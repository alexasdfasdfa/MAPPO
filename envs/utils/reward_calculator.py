import math
import numpy as np

from dataclasses import dataclass
from typing import Optional, Any

from envs.utils.utils import reach_goal, cal_distance


@dataclass(frozen=True)
class RewardContext:
    """
    Reward computation context for future extensibility.

    Note: current default reward implementation does not require most fields.
    """

    pattern_name: Optional[str] = None
    iteration_step: Optional[int] = None
    # You can extend this with more signals later.
    extra: Optional[Any] = None


class RewardCalculator:
    """
    Central place to compute rewards.

    - `compute_default_reward` uses non-dynamic shaping from config (`nd_*` args).
    - Dynamic mode adds optional Laplacian formation via `dynamic_discount_formation` × time schedule.
    - `compute_pattern_speed_direction_reward` is a placeholder for future reward shaping.
    """

    def __init__(self, env_core: Any):
        self.env = env_core
        self._dynamic_step_cache: Optional[dict[str, Any]] = None

    def _dynamic_v2_enabled(self, a: Any) -> bool:
        if int(getattr(a, "dynamic_reward_cluster_v2", 1)) == 0:
            return False
        if int(getattr(a, "dynamic_cluster_reward_requires_centralized_v", 1)) != 0:
            return bool(getattr(a, "use_centralized_V", False))
        return True

    def begin_dynamic_reward_step(self, env: Any, for_feature: float) -> None:
        if not getattr(env, "dynamic_goal_assignment", False):
            self._dynamic_step_cache = None
            return
        if not self._dynamic_v2_enabled(env.args):
            self._dynamic_step_cache = None
            return
        from envs.utils.dynamic_ctde_cluster import build_step_cache

        self._dynamic_step_cache = build_step_cache(env, for_feature)

    def _attn_comm_shaping_reward(self, robot: Any) -> tuple[float, dict[str, float]]:
        """
        Optional dense signals for attn_comm (fixed targets): encourage in-radius intent alignment,
        non-collapsed team messages, and smooth temporal updates. Scaled by attn_comm_reward_coef.
        """
        env = self.env
        if not getattr(env, "use_attn_comm_actor", False):
            return 0.0, {}
        a = env.args
        coef = float(getattr(a, "attn_comm_reward_coef", 0.0))
        if coef <= 0.0:
            return 0.0, {}
        try:
            ri = env.robots.index(robot)
        except ValueError:
            return 0.0, {}

        w_a = float(getattr(a, "attn_comm_reward_w_align", 1.0))
        w_d = float(getattr(a, "attn_comm_reward_w_diversity", 0.25))
        w_s = float(getattr(a, "attn_comm_reward_w_smooth", 0.08))

        msg = env.agent_broadcast_msg
        prev = env.agent_broadcast_msg_prev
        n = int(env.robot_num)
        d_m = int(msg.shape[1])
        Rc = float(env.attn_comm_radius)

        px, py = float(env.robots[ri].px), float(env.robots[ri].py)
        m_i = msg[ri]
        ni = float(np.linalg.norm(m_i)) + 1e-8
        cosines = []
        for j in range(n):
            if j == ri:
                continue
            rj = env.robots[j]
            if cal_distance(px, py, rj.px, rj.py) > Rc + 1e-9:
                continue
            m_j = msg[j]
            nj = float(np.linalg.norm(m_j)) + 1e-8
            cosines.append(float(np.dot(m_i, m_j) / (ni * nj)))
        align = float(np.mean(cosines)) if cosines else 0.0

        nm = msg / (np.linalg.norm(msg, axis=1, keepdims=True) + 1e-8)
        if n >= 2:
            dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    dists.append(float(np.linalg.norm(nm[i] - nm[j])))
            diversity = float(np.mean(dists))
        else:
            diversity = 0.0

        pn = float(np.linalg.norm(prev[ri]))
        if pn > 1e-5:
            smooth_term = -float(np.mean((msg[ri] - prev[ri]) ** 2)) / max(float(d_m), 1.0)
        else:
            smooth_term = 0.0

        raw = w_a * align + w_d * diversity + w_s * smooth_term
        r_comm = coef * raw
        terms = {
            "r_attn_comm_raw": float(raw),
            "c_attn_comm": float(r_comm),
            "attn_comm_align": float(align),
            "attn_comm_diversity": float(diversity),
            "attn_comm_smooth_term": float(smooth_term),
        }
        return r_comm, terms

    def _formation_time_weight(self) -> float:
        """Linear in τ_eff∈[0,1]; multiply formation base weights.

        Uses time at the start of the current transition (global_time was already
        advanced by time_step before reward), so the first step uses τ=0 → start weight.
        τ_eff = min(1, τ / decay_horizon): smaller horizon ⇒ faster decay toward t1.
        """
        a = self.env.args
        t0 = float(getattr(a, "formation_time_weight_start", 1.0))
        t1 = float(getattr(a, "formation_time_weight_end", 1.0))
        tl = max(float(getattr(self.env, "time_limit", 1.0)), 1e-9)
        dt = float(getattr(self.env, "time_step", 0.1))
        gt = float(getattr(self.env, "global_time", 0.0))
        t_elapsed = max(0.0, gt - dt)
        tau = min(1.0, t_elapsed / tl)
        h = float(getattr(a, "formation_time_weight_decay_horizon", 1.0))
        h = max(1e-6, min(1.0, h))
        tau_eff = min(1.0, tau / h)
        return t0 + (t1 - t0) * tau_eff

    def _nd_goal_disk_raw_reward(self, robot: Any, a: Any) -> tuple[float, dict[str, float]]:
        """
        Goal-disk shaping for static_nd / undetermined: arrival + per-step stay + velocity inside disk;
        subtract nd_goal_leave_penalty when the agent was inside the disk last state and is outside now.
        r_formation remains <= 0 elsewhere (penalty-only).
        """
        robot.goal_flag = False
        r_goal = 0.0
        terms: dict[str, float] = {"nd_goal_leave_penalty_raw": 0.0}
        rr = float(getattr(robot, "radius", 0.3))
        if reach_goal(robot):
            robot.goal_flag = True
            r_goal += float(getattr(a, "nd_arrival_reward", 0.0))
            r_goal += float(getattr(a, "nd_goal_stay_reward", 0.0))
            vb = float(getattr(a, "nd_goal_inside_velocity_bonus", 0.0))
            if float(getattr(robot, "v", 0.0)) > 1e-6:
                r_goal += vb
        else:
            lp = float(getattr(a, "nd_goal_leave_penalty", 0.0))
            if lp > 1e-12 and robot.pre_dist2goal is not None:
                pre_in = float(robot.pre_dist2goal) <= rr + 1e-9
                cur_out = float(robot.dist2goal) > rr + 1e-9
                if pre_in and cur_out:
                    r_goal -= lp
                    terms["nd_goal_leave_penalty_raw"] = float(lp)
        return r_goal, terms

    def _nd_progress_delta(self, robot: Any, a: Any) -> float:
        """Per-step distance change toward goal, clipped to limit nav exploit."""
        if robot.pre_dist2goal is None:
            return 0.0
        raw = float(robot.pre_dist2goal) - float(robot.dist2goal)
        clip = float(getattr(a, "nd_nav_progress_clip", 0.0))
        if clip > 1e-12:
            raw = max(-clip, min(clip, raw))
        return raw

    def _nd_timeout_no_goal_penalty(self, robot: Any, a: Any) -> tuple[float, dict[str, float]]:
        """Sparse penalty on the first step global_time crosses time_limit, if agent not at goal."""
        pen = float(getattr(a, "nd_timeout_no_goal_penalty", 0.0))
        if pen >= -1e-12:
            return 0.0, {}
        env = self.env
        tl = float(getattr(env, "time_limit", 1e9))
        gt = float(getattr(env, "global_time", 0.0))
        dt = float(getattr(env, "time_step", 0.1))
        if gt + 1e-9 < tl:
            return 0.0, {}
        if gt - dt + 1e-9 >= tl:
            return 0.0, {}
        if robot.collision is True:
            return 0.0, {}
        if reach_goal(robot):
            return 0.0, {}
        return pen, {"nd_timeout_no_goal_penalty_raw": float(pen)}

    def _dynamic_explore_undervisible_bonus(
        self, robot: Any, *, density_mult: float = 1.0
    ) -> tuple[float, dict[str, float]]:
        """
        Local M-slot packing: if some of the M nearest goals are outside R_vis, reward velocity aligned
        with the direction to the nearest such goal. density_mult (from local teammate count) scales this:
        high when crowded, low when sparse so agents prioritize reaching nearby targets first.
        """
        env = self.env
        a = env.args
        dm0 = max(0.0, float(density_mult))
        z = {
            "r_explore_undervisible_raw": 0.0,
            "dynamic_explore_shortfall": 0.0,
            "dynamic_explore_n_in_view": 0.0,
            "dynamic_explore_align": 0.0,
            "dynamic_explore_density_mult": float(dm0),
            "dynamic_explore_commit_suppressed": 0.0,
        }
        if not getattr(env, "dynamic_goal_assignment", False):
            return 0.0, z
        if str(getattr(env, "dynamic_obs_pack_version", "slots")) == "legacy":
            return 0.0, z
        scale = float(getattr(a, "dynamic_explore_undervisible_scale", 0.0))
        if scale <= 0.0:
            return 0.0, z
        if robot.collision is True or robot.success is True:
            return 0.0, z
        if self._dynamic_in_assigned_goal_commit_zone(robot):
            z["dynamic_explore_commit_suppressed"] = 1.0
            return 0.0, z
        M = int(env.dynamic_slot_m)
        K = int(env.num_goal_targets)
        if M < 1 or K < 1:
            return 0.0, z
        m_pack = min(M, K)
        Rv = float(env.dynamic_vis_radius)
        px, py = float(robot.px), float(robot.py)
        theta = float(robot.theta)
        dist_idx = []
        for k in range(K):
            tx, ty = env.goal_positions[k]
            dist_idx.append((cal_distance(px, py, tx, ty), k))
        dist_idx.sort(key=lambda t: t[0])
        n_in = 0
        outside: list[tuple[float, int, float, float]] = []
        for slot in range(m_pack):
            d, k = dist_idx[slot]
            tx, ty = env.goal_positions[k]
            gdx, gdy = float(tx - px), float(ty - py)
            if d <= Rv + 1e-9:
                n_in += 1
            else:
                outside.append((d, k, gdx, gdy))
        short = m_pack - n_in
        min_sf = max(1, int(getattr(a, "dynamic_explore_min_shortfall", 1)))
        z["dynamic_explore_n_in_view"] = float(n_in)
        z["dynamic_explore_shortfall"] = float(short)
        if short < min_sf or not outside:
            return 0.0, z
        req_unc = int(getattr(a, "dynamic_explore_require_unclaimed_outside", 0)) != 0
        cb = getattr(env, "claimed_by", None)
        if req_unc and cb is not None:
            outside = [t for t in outside if int(cb[t[1]]) < 0]
            if not outside:
                return 0.0, z
        outside.sort(key=lambda t: t[0])
        _, _, gdx, gdy = outside[0]
        ng = math.hypot(gdx, gdy)
        if ng < 1e-6:
            return 0.0, z
        hx, hy = math.cos(theta), math.sin(theta)
        align = (hx * gdx + hy * gdy) / ng
        z["dynamic_explore_align"] = float(align)
        vref = max(float(getattr(a, "dynamic_explore_v_ref", 1.0)), 1e-6)
        v = float(getattr(robot, "v", 0.0))
        w_short = float(short) / float(m_pack)
        raw = scale * dm0 * w_short * max(0.0, align) * min(v / vref, 1.0)
        z["r_explore_undervisible_raw"] = float(raw)
        return float(raw), z

    def _dynamic_in_assigned_goal_commit_zone(self, robot: Any) -> bool:
        """
        Near the current assignment (gx, gy): suppress rewards that encourage looking/sprinting toward *other* goals,
        which otherwise spike when neighbors crowd (full explore scale) or when same-target contest turns on flee shaping.
        """
        env = self.env
        if not getattr(env, "dynamic_goal_assignment", False):
            return False
        mult = float(getattr(env.args, "dynamic_shaping_commit_rvis_mult", 1.0))
        if mult <= 1e-9:
            return False
        if robot.collision or robot.success:
            return False
        Rv = float(getattr(env, "dynamic_vis_radius", 5.0))
        thr = mult * Rv
        d = getattr(robot, "dist2goal", None)
        if d is None:
            return False
        return float(d) <= thr + 1e-9

    def _dynamic_local_radius(self) -> float:
        a = self.env.args
        R = float(getattr(a, "dynamic_local_density_radius", 0.0))
        if R <= 1e-9:
            R = float(getattr(a, "dynamic_crowding_dist", 0.0))
        if R <= 1e-9:
            R = 3.0
        return R

    def _dynamic_local_active_neighbor_count(self, ridx: int) -> int:
        if ridx < 0:
            return 0
        env = self.env
        R = self._dynamic_local_radius()
        robots = env.robots
        if ridx >= len(robots):
            return 0
        ri = robots[ridx]
        if ri.collision or ri.success:
            return 0
        c = 0
        for j, rj in enumerate(robots):
            if j == ridx or rj.collision or rj.success:
                continue
            if cal_distance(ri.px, ri.py, rj.px, rj.py) <= R + 1e-9:
                c += 1
        return c

    def _dynamic_explore_density_multiplier(self, n_nb: int) -> float:
        a = self.env.args
        lo = int(getattr(a, "dynamic_explore_density_sparse_max", 1))
        hi = int(getattr(a, "dynamic_explore_density_dense_min", 3))
        lo = max(0, lo)
        hi = max(lo + 1, hi)
        mn = float(getattr(a, "dynamic_explore_low_density_mult", 0.12))
        mn = max(0.0, min(1.0, mn))
        if n_nb <= lo:
            return mn
        if n_nb >= hi:
            return 1.0
        t = (float(n_nb) - float(lo)) / float(hi - lo)
        return mn + (1.0 - mn) * t

    def _dynamic_sparse_urgency_multiplier(self, n_nb: int) -> float:
        a = self.env.args
        base = float(getattr(a, "dynamic_sparse_urgency_max_mult", 1.24))
        if base <= 1.0:
            return 1.0
        n_u = int(getattr(a, "dynamic_sparse_urgency_neighbors_max", 1))
        n_n = int(getattr(a, "dynamic_sparse_urgency_neighbors_norm", 4))
        n_u = max(0, n_u)
        n_n = max(n_u + 1, n_n)
        if n_nb <= n_u:
            return base
        if n_nb >= n_n:
            return 1.0
        t = (float(n_nb) - float(n_u)) / float(n_n - n_u)
        return base + (1.0 - base) * t

    def _dynamic_chasing_claimed_target(self, ridx: int, robot: Any, K: int) -> bool:
        if ridx < 0 or robot.collision or robot.success:
            return False
        cb = getattr(self.env, "claimed_by", None)
        if cb is None or len(cb) != K:
            return False
        tk = int(robot.target_id) % K
        owner = int(cb[tk])
        return owner >= 0 and owner != ridx

    def _dynamic_same_target_locally_contested(self, ridx: int, robot: Any, K: int) -> bool:
        """
        True if some active teammate within local_radius shares our target_id and is closer to that goal
        by more than dynamic_sparse_contest_margin (stronger local claim to the same target).
        """
        if ridx < 0 or robot.collision or robot.success:
            return False
        tid = int(robot.target_id) % K
        gx, gy = self.env.goal_positions[tid]
        d_self = cal_distance(robot.px, robot.py, gx, gy)
        margin = float(getattr(self.env.args, "dynamic_sparse_contest_margin", 0.15))
        R = self._dynamic_local_radius()
        robots = self.env.robots
        for j, rj in enumerate(robots):
            if j == ridx or rj.collision or rj.success:
                continue
            if cal_distance(robot.px, robot.py, rj.px, rj.py) > R + 1e-9:
                continue
            if int(rj.target_id) % K != tid:
                continue
            if cal_distance(rj.px, rj.py, gx, gy) < d_self - margin:
                return True
        return False

    def _dynamic_low_density_explore_bonus(
        self,
        robot: Any,
        ridx: int,
        *,
        active: bool,
        n_nb: int,
        chasing_claimed: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """
        Reward moving toward lower agent density when active (contested same-target or chasing claimed goal).
        With local neighbors: direction away from their centroid; otherwise sparsest forward sector.
        """
        z = {
            "r_low_density_explore_raw": 0.0,
            "dynamic_low_density_mode": 0.0,
            "dynamic_low_density_commit_suppressed": 0.0,
        }
        if not active:
            return 0.0, z
        a = self.env.args
        scale = float(getattr(a, "dynamic_low_density_explore_scale", 0.0))
        if scale <= 0.0:
            return 0.0, z
        if robot.collision or robot.success:
            return 0.0, z
        if self._dynamic_in_assigned_goal_commit_zone(robot) and not chasing_claimed:
            z["dynamic_low_density_commit_suppressed"] = 1.0
            return 0.0, z
        px, py = float(robot.px), float(robot.py)
        theta = float(robot.theta)
        vref = max(float(getattr(a, "dynamic_low_density_explore_v_ref", 1.0)), 1e-6)
        v = float(getattr(robot, "v", 0.0))
        robots = self.env.robots
        Rloc = self._dynamic_local_radius()

        gdx, gdy = 0.0, 0.0
        mode = 0.0
        if n_nb >= 1:
            sx, sy = 0.0, 0.0
            c = 0
            for j, rj in enumerate(robots):
                if j == ridx or rj.collision or rj.success:
                    continue
                if cal_distance(px, py, rj.px, rj.py) <= Rloc + 1e-9:
                    sx += float(rj.px)
                    sy += float(rj.py)
                    c += 1
            if c >= 1:
                cx, cy = sx / float(c), sy / float(c)
                gdx, gdy = px - cx, py - cy
                mode = 1.0

        if mode < 0.5 or math.hypot(gdx, gdy) < 1e-6:
            Rsec = float(getattr(a, "dynamic_low_density_sector_radius", 0.0))
            if Rsec <= 1e-9:
                Rsec = max(2.0 * Rloc, 8.0)
            cos_half = float(getattr(a, "dynamic_low_density_sector_cos", 0.707))
            nsect = max(4, int(getattr(a, "dynamic_low_density_sector_count", 8)))
            best_c = 10**9
            best_ux, best_uy = 1.0, 0.0
            for s in range(nsect):
                ang = (2.0 * math.pi * s) / float(nsect)
                ux, uy = math.cos(ang), math.sin(ang)
                cnt = 0
                for j, oj in enumerate(robots):
                    if j == ridx or oj.collision or oj.success:
                        continue
                    dx, dy = float(oj.px) - px, float(oj.py) - py
                    dist = math.hypot(dx, dy)
                    if dist <= 1e-6 or dist > Rsec:
                        continue
                    cs = (dx * ux + dy * uy) / dist
                    if cs >= cos_half:
                        cnt += 1
                if cnt < best_c:
                    best_c = cnt
                    best_ux, best_uy = ux, uy
            gdx, gdy = best_ux, best_uy
            mode = 2.0

        ng = math.hypot(gdx, gdy)
        if ng < 1e-9:
            return 0.0, z
        hx, hy = math.cos(theta), math.sin(theta)
        align = (hx * gdx + hy * gdy) / ng
        raw = scale * max(0.0, align) * min(v / vref, 1.0)
        z["dynamic_low_density_mode"] = float(mode)
        z["r_low_density_explore_raw"] = float(raw)
        return float(raw), z

    def _dynamic_find_reciprocal_swap_indices(self) -> set[int]:
        """
        Indices of agents that (this step) exchanged target_ids with a partner and both
        head roughly toward each other (counter-flow / de-conflict via swap).
        """
        env = self.env
        a = env.args
        sw_scale = float(getattr(a, "dynamic_reciprocal_swap_reward_scale", 0.0))
        swm = float(getattr(a, "dynamic_reciprocal_swap_switch_penalty_mult", 1.0))
        if sw_scale <= 0.0 and swm >= 1.0 - 1e-9:
            return set()
        if not getattr(env, "dynamic_goal_assignment", False):
            return set()
        K = int(env.num_goal_targets)
        if K < 2:
            return set()
        cos_th = float(getattr(a, "dynamic_swap_approach_cos_thresh", 0.12))
        robots = env.robots
        n = len(robots)
        out: set[int] = set()
        for i in range(n):
            ri = robots[i]
            if ri.collision or ri.success:
                continue
            for j in range(i + 1, n):
                rj = robots[j]
                if rj.collision or rj.success:
                    continue
                oi = int(ri.prev_target_id) % K
                oj = int(rj.prev_target_id) % K
                ni = int(ri.target_id) % K
                nj = int(rj.target_id) % K
                if oi == oj:
                    continue
                if not (
                    bool(getattr(ri, "target_switched_this_step", False))
                    and bool(getattr(rj, "target_switched_this_step", False))
                ):
                    continue
                if ni != oj or nj != oi:
                    continue
                dx = float(rj.px - ri.px)
                dy = float(rj.py - ri.py)
                s2 = dx * dx + dy * dy
                if s2 < 1e-8:
                    continue
                inv = 1.0 / math.sqrt(s2)
                ux, uy = dx * inv, dy * inv
                hi = (math.cos(float(ri.theta)), math.sin(float(ri.theta)))
                hj = (math.cos(float(rj.theta)), math.sin(float(rj.theta)))
                ai = hi[0] * ux + hi[1] * uy
                aj = hj[0] * (-ux) + hj[1] * (-uy)
                if ai < cos_th or aj < cos_th:
                    continue
                out.add(i)
                out.add(j)
        return out

    def _dynamic_goal_contention_penalty(self) -> tuple[float, float]:
        """
        Shared-style deduction (already ÷n inside): sum over goals of max(0, c-1) for agents
        assigned that target and within radius of the goal center.
        Returns (penalty_chunk_to_subtract_from_shared, raw_excess_agent_count_sum).
        """
        env = self.env
        a = env.args
        scale = float(getattr(a, "dynamic_goal_contention_penalty_scale", 0.0))
        if scale <= 0.0:
            return 0.0, 0.0
        if not getattr(env, "dynamic_goal_assignment", False):
            return 0.0, 0.0
        K = int(env.num_goal_targets)
        n = max(1, env.robot_num)
        robots = env.robots
        R = float(getattr(a, "dynamic_goal_contention_radius", 0.0))
        if R <= 1e-9:
            rr = float(getattr(robots[0], "radius", 0.3)) if robots else 0.3
            R = max(2.0 * rr, 0.5)
        excess_sum = 0
        for k in range(K):
            gx, gy = env.goal_positions[k]
            cnt = 0
            for r in robots:
                if r.collision or r.success:
                    continue
                if int(r.target_id) % K != k:
                    continue
                if cal_distance(r.px, r.py, gx, gy) <= R + 1e-9:
                    cnt += 1
            if cnt > 1:
                excess_sum += cnt - 1
        pen = scale * float(excess_sum) / float(n)
        return pen, float(excess_sum)

    @staticmethod
    def _hausdorff_directed(A: np.ndarray, B: np.ndarray) -> float:
        """h(A,B) = max_{x in A} min_{y in B} ||x-y|| (2D)."""
        if A.size == 0 or B.size == 0:
            return 0.0
        worst = 0.0
        for i in range(A.shape[0]):
            d = np.sqrt(((B - A[i]) ** 2).sum(axis=1))
            worst = max(worst, float(np.min(d)))
        return float(worst)

    def _hausdorff_2d(self, P: np.ndarray, Q: np.ndarray) -> float:
        """Symmetric Hausdorff distance d_HD(P,Q) = max(h(P,Q), h(Q,P))."""
        if P.shape[0] == 0 or Q.shape[0] == 0:
            return 0.0
        return max(self._hausdorff_directed(P, Q), self._hausdorff_directed(Q, P))

    def _ensure_attn_undetermined_team_cache(self) -> dict:
        """
        Paper (arXiv:2307.12287) Eq.(1)-(4): formation (HD + lag), navigation (centroid-destination + lag),
        collision count; shared team terms, cached once per env step.
        """
        env = self.env
        key = float(getattr(env, "global_time", 0.0))
        if getattr(self, "_aud_team_key", None) == key:
            return self._aud_team  # type: ignore
        self._aud_team_key = key
        a = env.args
        n = int(env.robot_num)
        pos = np.array([[float(r.px), float(r.py)] for r in env.robots], dtype=np.float64)
        gp = getattr(env, "goal_positions", None)
        if gp is None or len(gp) < n:
            self._aud_team = {
                "rf": 0.0,
                "rv": 0.0,
                "rc": 0.0,
                "r_team": 0.0,
                "d_hd": 0.0,
                "centroid_dist": 0.0,
                "n_coll_pairs": 0.0,
            }
            return self._aud_team
        goals = np.array([[float(t[0]), float(t[1])] for t in gp[:n]], dtype=np.float64)
        c_pos = pos.mean(axis=0)
        P = pos - c_pos
        c_goal = goals.mean(axis=0)
        Delta = goals - c_goal
        rp = float(np.linalg.norm(P, axis=1).mean()) + 1e-6
        rd = float(np.linalg.norm(Delta, axis=1).mean()) + 1e-6
        Delta_scaled = Delta * (rp / rd)
        d_hd = self._hausdorff_2d(P, Delta_scaled)
        w1 = float(getattr(a, "cons_decaf_omega1_lag", 0.1))
        w2 = float(getattr(a, "cons_decaf_omega2_lag", 0.1))
        rf = -d_hd - w1 * float(getattr(env, "_cons_decaf_rf_prev", 0.0))
        env._cons_decaf_rf_prev = rf
        nav_d = float(np.linalg.norm(c_pos - c_goal))
        rv = -nav_d - w2 * float(getattr(env, "_cons_decaf_rv_prev", 0.0))
        env._cons_decaf_rv_prev = rv
        rr = float(env.robots[0].radius) if env.robots else 0.35
        delta_safe = float(getattr(a, "cons_decaf_delta_safe", 0.0))
        if delta_safe <= 1e-9:
            delta_safe = 2.0 * rr
        n_coll = 0
        for i in range(n):
            for j in range(i + 1, n):
                if cal_distance(env.robots[i].px, env.robots[i].py, env.robots[j].px, env.robots[j].py) < delta_safe:
                    n_coll += 1
        rc = float(n_coll)
        wf = float(getattr(a, "cons_decaf_omega_f", 1.0))
        wv = float(getattr(a, "cons_decaf_omega_v", 1.0))
        wc = float(getattr(a, "cons_decaf_omega_c", 1.0))
        r_team = wf * rf + wv * rv - wc * rc
        self._aud_team = {
            "rf": float(rf),
            "rv": float(rv),
            "rc": float(rc),
            "r_team": float(r_team),
            "d_hd": float(d_hd),
            "centroid_dist": float(nav_d),
            "n_coll_pairs": float(n_coll),
            "wf": wf,
            "wv": wv,
            "wc": wc,
        }
        return self._aud_team

    def _compute_attn_undetermined_goal_reward(self, robot: Any, for_feature: float) -> np.ndarray:
        """Undetermined execution + ConsMAC obs; reward from paper Eq.(1)-(4), identical per agent."""
        env = self.env
        a = env.args
        team = self._ensure_attn_undetermined_team_cache()
        reward = float(team["r_team"])
        r_sl, sl_v, sl_succ, sl_delta = self._undetermined_v2_sl_reward_raw(env, a)
        reward += r_sl
        gcoef = float(getattr(a, "cons_decaf_goal_disk_coef", 0.0))
        r_goal_add = 0.0
        nd_goal_terms: dict[str, float] = {}
        if gcoef > 0.0:
            r_goal_add, nd_goal_terms = self._nd_goal_disk_raw_reward(robot, a)
            reward += gcoef * float(getattr(a, "nd_discount_goal", 200.0)) * float(r_goal_add)

        tr = float(getattr(a, "nd_goal_terminal_reward", 0.0))
        entered = False
        nd_tr_applied = 0.0
        if robot.goal_flag:
            entered = bool(
                robot.pre_dist2goal is not None
                and robot.pre_dist2goal > robot.radius + 1e-9
            )
            if tr != 0.0 and entered:
                reward += tr
                nd_tr_applied = tr

        robot._reward_terms = {
            "reward_mode": "attn_undetermined_goal",
            "laplacian_S_L": float(sl_v),
            "undetermined_v2_sl_raw": float(r_sl),
            "undetermined_v2_sl_delta_raw": float(sl_delta),
            "undetermined_v2_sl_success": float(sl_succ),
            "cons_decaf_rf": float(team["rf"]),
            "cons_decaf_rv": float(team["rv"]),
            "cons_decaf_rc": float(team["rc"]),
            "cons_decaf_r_team": float(team["r_team"]),
            "cons_decaf_d_hd": float(team["d_hd"]),
            "cons_decaf_centroid_dist": float(team["centroid_dist"]),
            "cons_decaf_n_coll_pairs": float(team["n_coll_pairs"]),
            "r_goal_raw": float(r_goal_add),
            "reward_final": float(reward),
            "nd_terminal_tr": float(tr),
            "nd_tr_applied": float(nd_tr_applied),
            "goal_entered_from_outside": float(1.0 if entered else 0.0),
            "goal_flag": float(1.0 if robot.goal_flag else 0.0),
            **{k: float(v) for k, v in nd_goal_terms.items()},
        }
        return np.array([reward])

    def compute_default_reward(self, robot: Any, for_feature: float) -> np.ndarray:
        """
        Preserve EnvCore.get_reward() current logic.
        """
        if getattr(self.env, "dynamic_goal_assignment", False):
            return self._compute_dynamic_goal_reward(robot, for_feature)
        if getattr(self.env, "undetermined_goal_assignment", False):
            _a = self.env.args
            if (
                getattr(self.env, "undetermined_goal_v2", False)
                or getattr(self.env, "undetermined_goal_v3", False)
            ) and str(
                getattr(_a, "architecture_mode", "default")
            ) == "attn_undetermined_goal":
                return self._compute_attn_undetermined_goal_reward(robot, for_feature)
            if getattr(self.env, "undetermined_goal_v2", False) or getattr(
                self.env, "undetermined_goal_v3", False
            ):
                return self._compute_undetermined_reward_v2(robot, for_feature)
            return self._compute_undetermined_reward(robot, for_feature)
        a = self.env.args
        r_avoid = 0
        r_goal = 0
        r_nav = 0
        r_bonus = 0
        # collision
        if robot.collision == True:
            r_avoid = -60
        else:
            if robot.dmin < robot.discomfort_dist * 2:
                r_avoid = -np.exp(-robot.dmin / 3)

        # formation
        r_formation = -np.sqrt(for_feature)

        # bonus (currently not used in final reward)
        if abs(robot.pre_theta - robot.theta) > 0.7:
            r_bonus = -1

        # navigation: distance progress (local-obs-friendly dense signal)
        prog_coef = float(getattr(a, "nd_goal_progress_coef", 5.0))
        prog_delta = self._nd_progress_delta(robot, a)
        r_nav += prog_delta * prog_coef

        pr = float(getattr(a, "nd_proximity_reward_scale", 0.0))
        if pr != 0.0:
            sig = max(float(getattr(a, "nd_proximity_sigma", 10.0)), 1e-6)
            r_nav += pr * float(np.exp(-robot.dist2goal / sig))

        hs = float(getattr(a, "nd_heading_reward_scale", 0.0))
        if hs != 0.0 and robot.v > 1e-6:
            gdx = float(robot.gx - robot.px)
            gdy = float(robot.gy - robot.py)
            ng = float(np.hypot(gdx, gdy))
            if ng > 1e-6:
                c = (np.cos(robot.theta) * gdx + np.sin(robot.theta) * gdy) / ng
                vref = max(float(getattr(a, "nd_heading_v_ref", 1.0)), 1e-6)
                r_nav += hs * max(0.0, float(c)) * min(float(robot.v) / vref, 1.0)

        if r_nav > 0 and robot.collision:
            r_nav = 0

        tpen, tterms = self._nd_timeout_no_goal_penalty(robot, a)
        r_nav += tpen

        r_goal_add, nd_goal_terms = self._nd_goal_disk_raw_reward(robot, a)
        r_goal += r_goal_add

        if self.env.collision_flag:
            r_goal = 0

        discount_formation = float(getattr(a, "nd_discount_formation", 0.0))
        w_form = self._formation_time_weight()
        discount_avoid = float(getattr(a, "nd_discount_avoid", 50.0))
        discount_nav = float(getattr(a, "nd_discount_nav", 20.0))
        discount_goal = float(getattr(a, "nd_discount_goal", 200.0))
        discount_bonus = 2

        c_formation = discount_formation * w_form * r_formation
        c_avoid = discount_avoid * r_avoid
        c_nav = discount_nav * r_nav
        c_goal = discount_goal * r_goal

        reward_shaped_nd = c_formation + c_avoid + c_nav + c_goal
        r_comm, comm_terms = self._attn_comm_shaping_reward(robot)
        reward_mid = reward_shaped_nd + r_comm
        reward_shaped = reward_mid
        reward = reward_mid
        tr = float(getattr(a, "nd_goal_terminal_reward", 0.0))
        entered = False
        nd_tr_applied = 0.0
        if robot.goal_flag:
            entered = bool(
                robot.pre_dist2goal is not None
                and robot.pre_dist2goal > robot.radius + 1e-9
            )
            if tr != 0.0 and entered:
                reward = reward_mid + tr
                nd_tr_applied = tr
            else:
                reward = reward_mid

        robot._reward_terms = {
            "reward_mode": "static_nd",
            "r_avoid_raw": float(r_avoid),
            "r_formation_raw": float(r_formation),
            "r_nav_raw": float(r_nav),
            "r_goal_raw": float(r_goal),
            "r_bonus_raw": float(r_bonus),
            "nd_progress_delta_applied": float(prog_delta),
            "c_formation": float(c_formation),
            "formation_time_w": float(w_form),
            "c_avoid": float(c_avoid),
            "c_nav": float(c_nav),
            "c_goal": float(c_goal),
            "reward_shaped": float(reward_shaped),
            "nd_terminal_tr": float(tr),
            "nd_tr_applied": float(nd_tr_applied),
            "goal_entered_from_outside": float(1.0 if entered else 0.0),
            "goal_flag": float(1.0 if robot.goal_flag else 0.0),
            "reward_final": float(reward),
            **{k: float(v) for k, v in nd_goal_terms.items()},
            **{k: float(v) for k, v in tterms.items()},
            **{k: float(v) for k, v in comm_terms.items()},
        }

        return np.array([reward])

    def _compute_undetermined_reward(self, robot: Any, for_feature: float) -> np.ndarray:
        """
        Like non-dynamic (nd_*) shaping + one-time Hungarian gap bonus per apply_undetermined_targets.
        Extra: when far from assigned (gx,gy), progress coef is boosted; optional -scale*dist2goal pulls
        agents in world frame (reduces drifting in relative S-shape without entering targets).
        """
        env = self.env
        a = env.args
        ridx = 0
        try:
            ridx = env.robots.index(robot)
        except ValueError:
            pass

        r_avoid = 0.0
        r_goal = 0.0
        r_nav = 0.0
        r_bonus = 0.0
        if robot.collision == True:
            r_avoid = -60.0
        else:
            if robot.dmin < robot.discomfort_dist * 2:
                r_avoid = -float(np.exp(-robot.dmin / 3))

        r_formation = -np.sqrt(max(float(for_feature), 0.0))
        if abs(robot.pre_theta - robot.theta) > 0.7:
            r_bonus = -1.0

        prog_coef = float(getattr(a, "nd_goal_progress_coef", 5.0))
        pc = prog_coef
        if robot.pre_dist2goal is not None and robot.dist2goal is not None:
            d_th = float(getattr(a, "undetermined_far_goal_progress_dist_thresh", 4.0))
            bst = float(getattr(a, "undetermined_far_goal_progress_boost", 1.5))
            if float(robot.dist2goal) > d_th:
                pc *= bst
        prog_delta = self._nd_progress_delta(robot, a)
        r_nav += prog_delta * pc

        pdp = float(getattr(a, "undetermined_goal_distance_penalty_scale", 0.0))
        und_dist_pen = 0.0
        if pdp > 1e-12 and robot.dist2goal is not None:
            und_dist_pen = pdp * float(robot.dist2goal)
            r_nav -= und_dist_pen

        und_dist_quad = 0.0
        pdq = float(getattr(a, "undetermined_goal_dist_penalty_quad_scale", 0.0))
        if pdq > 1e-12 and robot.dist2goal is not None:
            dg = float(robot.dist2goal)
            und_dist_quad = pdq * dg * dg
            r_nav -= und_dist_quad

        pr = float(getattr(a, "nd_proximity_reward_scale", 0.0))
        if pr != 0.0:
            sig = max(float(getattr(a, "nd_proximity_sigma", 10.0)), 1e-6)
            r_nav += pr * float(np.exp(-robot.dist2goal / sig))

        hs = float(getattr(a, "nd_heading_reward_scale", 0.0))
        if hs != 0.0 and robot.v > 1e-6:
            gdx = float(robot.gx - robot.px)
            gdy = float(robot.gy - robot.py)
            ng = float(np.hypot(gdx, gdy))
            if ng > 1e-6:
                c = (np.cos(robot.theta) * gdx + np.sin(robot.theta) * gdy) / ng
                vref = max(float(getattr(a, "nd_heading_v_ref", 1.0)), 1e-6)
                r_nav += hs * max(0.0, float(c)) * min(float(robot.v) / vref, 1.0)

        if r_nav > 0 and robot.collision:
            r_nav = 0.0

        tpen, tterms = self._nd_timeout_no_goal_penalty(robot, a)
        r_nav += tpen

        r_goal_add, nd_goal_terms = self._nd_goal_disk_raw_reward(robot, a)
        r_goal += r_goal_add

        if env.collision_flag:
            r_goal = 0.0

        discount_formation = float(getattr(a, "nd_discount_formation", 0.0))
        w_form = self._formation_time_weight()
        discount_avoid = float(getattr(a, "nd_discount_avoid", 50.0))
        discount_nav = float(getattr(a, "nd_discount_nav", 20.0))
        discount_goal = float(getattr(a, "nd_discount_goal", 200.0))

        c_formation = discount_formation * w_form * r_formation
        c_avoid = discount_avoid * r_avoid
        c_nav = discount_nav * r_nav
        c_goal = discount_goal * r_goal

        hung = 0.0
        if hasattr(env, "undetermined_hungarian_bonus") and env.undetermined_hungarian_bonus is not None:
            if 0 <= ridx < len(env.undetermined_hungarian_bonus):
                hung = float(env.undetermined_hungarian_bonus[ridx])
                env.undetermined_hungarian_bonus[ridx] = 0.0

        reward_mid = c_formation + c_avoid + c_nav + c_goal
        reward_shaped_nd = reward_mid
        reward = reward_mid + hung

        tr = float(getattr(a, "nd_goal_terminal_reward", 0.0))
        entered = False
        nd_tr_applied = 0.0
        if robot.goal_flag:
            entered = bool(
                robot.pre_dist2goal is not None
                and robot.pre_dist2goal > robot.radius + 1e-9
            )
            if tr != 0.0 and entered:
                reward = reward_shaped_nd + hung + tr
                nd_tr_applied = tr
            else:
                reward = reward_shaped_nd + hung

        robot._reward_terms = {
            "reward_mode": "undetermined",
            "r_avoid_raw": float(r_avoid),
            "r_formation_raw": float(r_formation),
            "r_nav_raw": float(r_nav),
            "r_goal_raw": float(r_goal),
            "nd_progress_delta_applied": float(prog_delta),
            "undetermined_dist_penalty_raw": float(und_dist_pen),
            "undetermined_dist_penalty_quad_raw": float(und_dist_quad),
            "c_formation": float(c_formation),
            "formation_time_w": float(w_form),
            "c_avoid": float(c_avoid),
            "c_nav": float(c_nav),
            "c_goal": float(c_goal),
            "undetermined_hungarian_bonus": float(hung),
            "reward_shaped": float(reward_shaped_nd),
            "reward_final": float(reward),
            "nd_terminal_tr": float(tr),
            "nd_tr_applied": float(nd_tr_applied),
            "goal_entered_from_outside": float(1.0 if entered else 0.0),
            "goal_flag": float(1.0 if robot.goal_flag else 0.0),
            **{k: float(v) for k, v in nd_goal_terms.items()},
            **{k: float(v) for k, v in tterms.items()},
        }

        return np.array([reward])

    def _undetermined_v2_sl_reward_raw(self, env: Any, a: Any) -> tuple[float, float, float, float]:
        """
        Type-2 (formation / similarity success): S_L = 1 - ||L_hat - L_des||_F / ||L_des||_F (EnvCore.laplacian_S_L).
        Dense term encourages high S_L; optional delta term rewards improvement vs previous step (pattern dynamics).
        L_des is built from goal sites so targets act mainly as a topological hint for the desired Laplacian.
        """
        scale_succ = float(getattr(a, "undetermined_v2_sl_success_scale", 0.0))
        scale_dense = float(getattr(a, "undetermined_v2_sl_dense_scale", 0.0))
        scale_delta = float(getattr(a, "undetermined_v2_sl_delta_scale", 0.0))
        thr = float(getattr(a, "undetermined_v2_sl_success_threshold", 0.97))
        sl = getattr(env, "laplacian_S_L", float("nan"))
        try:
            sl = float(sl)
        except (TypeError, ValueError):
            return 0.0, float("nan"), 0.0, 0.0
        if not math.isfinite(sl):
            return 0.0, sl, 0.0, 0.0
        pssh = float(getattr(a, "undetermined_v2_sl_post_success_sl_shaping_scale", 1.0))
        sl_shaping_w = pssh if (sl >= thr and pssh < 0.999) else 1.0
        r = 0.0
        if scale_dense > 1e-12:
            r += scale_dense * max(0.0, min(1.0, sl)) * sl_shaping_w
        sl_prev = getattr(env, "laplacian_S_L_prev", float("nan"))
        d_sl = 0.0
        sp = float("nan")
        try:
            sp = float(sl_prev)
            if math.isfinite(sp):
                d_sl = max(0.0, sl - sp)
        except (TypeError, ValueError):
            d_sl = 0.0
            sp = float("nan")
        r_delta = 0.0
        if scale_delta > 1e-12 and d_sl > 1e-12:
            r_delta = scale_delta * d_sl * sl_shaping_w
            r += r_delta
        succ = 1.0 if sl >= thr else 0.0
        crossing = sl >= thr and (not math.isfinite(sp) or sp < thr)
        only_cross = bool(getattr(a, "undetermined_v2_sl_success_only_on_crossing", False))
        sustain = float(getattr(a, "undetermined_v2_sl_success_sustain_frac", 0.0))
        if scale_succ > 1e-12 and sl >= thr:
            if only_cross:
                if crossing:
                    r += scale_succ
                elif sustain > 1e-12:
                    r += scale_succ * sustain
            else:
                r += scale_succ
        return r, sl, succ, float(r_delta)

    def _compute_undetermined_reward_v2(self, robot: Any, for_feature: float) -> np.ndarray:
        """
        Undetermined v2: same structure as v1 with capped soft-collision signal, approach shaping toward (gx,gy),
        and distance penalties already scaled by config floors + undetermined_v2_dist_penalty_mult.
        """
        env = self.env
        a = env.args
        ridx = 0
        try:
            ridx = env.robots.index(robot)
        except ValueError:
            pass

        r_avoid = 0.0
        r_goal = 0.0
        r_nav = 0.0
        r_bonus = 0.0
        cap_av = float(getattr(a, "undetermined_v2_avoid_exp_cap", 2.0))
        if robot.collision == True:
            r_avoid = -60.0
        else:
            if robot.dmin < robot.discomfort_dist * 2:
                r_avoid = -min(float(np.exp(-robot.dmin / 3)), max(cap_av, 1e-6))

        r_formation = -np.sqrt(max(float(for_feature), 0.0))
        if abs(robot.pre_theta - robot.theta) > 0.7:
            r_bonus = -1.0

        prog_coef = float(getattr(a, "nd_goal_progress_coef", 5.0))
        pc = prog_coef
        if robot.pre_dist2goal is not None and robot.dist2goal is not None:
            d_th = float(getattr(a, "undetermined_far_goal_progress_dist_thresh", 4.0))
            bst = float(getattr(a, "undetermined_far_goal_progress_boost", 1.5))
            if float(robot.dist2goal) > d_th:
                pc *= bst
        prog_delta = self._nd_progress_delta(robot, a)
        prog_nav = prog_delta * pc

        apr_s = float(getattr(a, "undetermined_v2_approach_reward_scale", 0.0))
        apr_cap = float(getattr(a, "undetermined_v2_approach_reward_cap", 0.55))
        approach_term = 0.0
        if (
            apr_s > 1e-9
            and not robot.collision
            and robot.pre_dist2goal is not None
            and robot.dist2goal is not None
        ):
            raw = float(robot.pre_dist2goal) - float(robot.dist2goal)
            approach_term = apr_s * min(max(raw, 0.0), apr_cap)

        pdp = float(getattr(a, "undetermined_goal_distance_penalty_scale", 0.0))
        und_dist_pen = 0.0
        if pdp > 1e-12 and robot.dist2goal is not None:
            und_dist_pen = pdp * float(robot.dist2goal)

        und_dist_quad = 0.0
        pdq = float(getattr(a, "undetermined_goal_dist_penalty_quad_scale", 0.0))
        if pdq > 1e-12 and robot.dist2goal is not None:
            dg = float(robot.dist2goal)
            und_dist_quad = pdq * dg * dg

        prox_term = 0.0
        pr = float(getattr(a, "nd_proximity_reward_scale", 0.0))
        if pr != 0.0:
            sig = max(float(getattr(a, "nd_proximity_sigma", 10.0)), 1e-6)
            prox_term = pr * float(np.exp(-robot.dist2goal / sig))

        head_term = 0.0
        hs = float(getattr(a, "nd_heading_reward_scale", 0.0))
        if hs != 0.0 and robot.v > 1e-6:
            gdx = float(robot.gx - robot.px)
            gdy = float(robot.gy - robot.py)
            ng = float(np.hypot(gdx, gdy))
            if ng > 1e-6:
                c = (np.cos(robot.theta) * gdx + np.sin(robot.theta) * gdy) / ng
                vref = max(float(getattr(a, "nd_heading_v_ref", 1.0)), 1e-6)
                head_term = hs * max(0.0, float(c)) * min(float(robot.v) / vref, 1.0)

        r_sl, sl_v, sl_succ, sl_delta = self._undetermined_v2_sl_reward_raw(env, a)
        thr_sl = float(getattr(a, "undetermined_v2_sl_success_threshold", 0.97))
        in_sl_succ = math.isfinite(sl_v) and sl_v >= thr_sl
        lit_arg = float(getattr(a, "undetermined_v2_sl_post_success_literal_scale", 1.0))
        lit_apply = lit_arg if in_sl_succ and lit_arg < 0.999 else 1.0
        pull_xy = prog_nav + approach_term - und_dist_pen - und_dist_quad + prox_term + head_term
        pull_xy *= lit_apply
        r_nav = pull_xy + r_sl

        pre_succ = math.isfinite(sl_v) and sl_v < thr_sl
        p_step = float(getattr(a, "undetermined_v2_sl_pre_success_step_penalty", 0.0))
        if p_step != 0.0 and pre_succ:
            r_nav += p_step
        p_tr = float(getattr(a, "undetermined_v2_sl_pre_success_travel_penalty", 0.0))
        if p_tr != 0.0 and pre_succ and hasattr(env, "dynamic_step_travel_sum"):
            r_nav += p_tr * float(env.dynamic_step_travel_sum) / max(1, int(env.robot_num))

        if r_nav > 0 and robot.collision:
            r_nav = 0.0

        tpen, tterms = self._nd_timeout_no_goal_penalty(robot, a)
        r_nav += tpen

        r_goal_add, nd_goal_terms = self._nd_goal_disk_raw_reward(robot, a)
        r_goal_add *= lit_apply
        r_goal += r_goal_add

        if env.collision_flag:
            r_goal = 0.0

        discount_formation = float(getattr(a, "nd_discount_formation", 0.0))
        w_form = self._formation_time_weight()
        discount_avoid = float(getattr(a, "nd_discount_avoid", 50.0))
        discount_nav = float(getattr(a, "nd_discount_nav", 20.0))
        discount_goal = float(getattr(a, "nd_discount_goal", 200.0))

        c_formation = discount_formation * w_form * r_formation
        c_avoid = discount_avoid * r_avoid
        c_nav = discount_nav * r_nav
        c_goal = discount_goal * r_goal

        hung = 0.0
        if hasattr(env, "undetermined_hungarian_bonus") and env.undetermined_hungarian_bonus is not None:
            if 0 <= ridx < len(env.undetermined_hungarian_bonus):
                hung = float(env.undetermined_hungarian_bonus[ridx])
                env.undetermined_hungarian_bonus[ridx] = 0.0

        reward_mid = c_formation + c_avoid + c_nav + c_goal
        reward_shaped_nd = reward_mid
        reward = reward_mid + hung

        tr = float(getattr(a, "nd_goal_terminal_reward", 0.0))
        entered = False
        nd_tr_applied = 0.0
        if robot.goal_flag:
            entered = bool(
                robot.pre_dist2goal is not None
                and robot.pre_dist2goal > robot.radius + 1e-9
            )
            if tr != 0.0 and entered:
                reward = reward_shaped_nd + hung + tr
                nd_tr_applied = tr
            else:
                reward = reward_shaped_nd + hung

        sl_prev_nf = float("nan")
        try:
            sl_prev_nf = float(getattr(env, "laplacian_S_L_prev", float("nan")))
        except (TypeError, ValueError):
            sl_prev_nf = float("nan")
        sl_crossing = (
            1.0
            if (
                math.isfinite(sl_v)
                and sl_v >= thr_sl
                and (not math.isfinite(sl_prev_nf) or sl_prev_nf < thr_sl)
            )
            else 0.0
        )
        pre_step_raw = float(p_step) if (p_step != 0.0 and pre_succ) else 0.0
        pre_tr_raw = (
            float(p_tr) * float(env.dynamic_step_travel_sum) / max(1, int(env.robot_num))
            if (p_tr != 0.0 and pre_succ and hasattr(env, "dynamic_step_travel_sum"))
            else 0.0
        )

        robot._reward_terms = {
            "reward_mode": "undetermined_v2",
            "r_avoid_raw": float(r_avoid),
            "r_formation_raw": float(r_formation),
            "r_nav_raw": float(r_nav),
            "r_goal_raw": float(r_goal),
            "nd_progress_delta_applied": float(prog_delta),
            "undetermined_v2_approach_raw": float(approach_term),
            "laplacian_S_L": float(sl_v),
            "undetermined_v2_sl_raw": float(r_sl),
            "undetermined_v2_sl_delta_raw": float(sl_delta),
            "undetermined_v2_sl_success": float(sl_succ),
            "undetermined_v2_sl_type2_crossing": float(sl_crossing),
            "undetermined_v2_sl_literal_relax_w": float(lit_apply),
            "undetermined_v2_sl_pre_success_step_raw": float(pre_step_raw),
            "undetermined_v2_sl_pre_success_travel_raw": float(pre_tr_raw),
            "undetermined_dist_penalty_raw": float(und_dist_pen),
            "undetermined_dist_penalty_quad_raw": float(und_dist_quad),
            "c_formation": float(c_formation),
            "formation_time_w": float(w_form),
            "c_avoid": float(c_avoid),
            "c_nav": float(c_nav),
            "c_goal": float(c_goal),
            "undetermined_hungarian_bonus": float(hung),
            "reward_shaped": float(reward_shaped_nd),
            "reward_final": float(reward),
            "nd_terminal_tr": float(tr),
            "nd_tr_applied": float(nd_tr_applied),
            "goal_entered_from_outside": float(1.0 if entered else 0.0),
            "goal_flag": float(1.0 if robot.goal_flag else 0.0),
            **{k: float(v) for k, v in nd_goal_terms.items()},
            **{k: float(v) for k, v in tterms.items()},
        }

        return np.array([reward])

    def _compute_dynamic_goal_reward(self, robot: Any, for_feature: float) -> np.ndarray:
        """Team travel, conflicts, coordinated distance + assignment shaping, arrival signal.

        Anti-exploit: same-target penalty is stronger when agents are crowded; optional arrival
        gate (env) requires entering from outside the goal disk; target switches near goals cost
        extra; r_nav is suppressed on target switches (gx,gy discontinuity).

        Local density: many nearby teammates → stronger explore outside R_vis; few neighbors →
        higher nav/prox urgency only when no stronger local competitor for the same target; chasing a
        claimed goal or losing a same-target race → penalty plus low-density exploration (toward fewer agents).
        """
        a = self.env.args
        n = max(1, self.env.robot_num)
        try:
            ridx = self.env.robots.index(robot)
        except ValueError:
            ridx = -1
        n_nb = self._dynamic_local_active_neighbor_count(ridx)
        K = int(self.env.num_goal_targets)
        contested = self._dynamic_same_target_locally_contested(ridx, robot, K)
        chasing_claimed = self._dynamic_chasing_claimed_target(ridx, robot, K)
        use_v2 = self._dynamic_v2_enabled(a)
        cache = self._dynamic_step_cache if use_v2 else None
        explore_dm = self._dynamic_explore_density_multiplier(n_nb)
        if use_v2:
            explore_dm = 0.0

        shared = 0.0
        shared -= float(getattr(a, "dynamic_path_reward_scale", 1.0)) * self.env.dynamic_step_travel_sum / n

        swap_idx = self._dynamic_find_reciprocal_swap_indices()
        cont_pen, cont_excess = self._dynamic_goal_contention_penalty()
        shared -= cont_pen

        in_swap = ridx >= 0 and ridx in swap_idx

        d_thr = float(getattr(a, "dynamic_same_target_conflict_dist", 2.0))
        pen_scale = float(getattr(a, "dynamic_same_target_penalty_scale", 2.0))
        if use_v2:
            pen_scale *= float(getattr(a, "dynamic_v2_shared_conflict_mult", 1.0))
        Dc = float(getattr(a, "dynamic_crowding_dist", 0.0))
        cl_boost = float(getattr(a, "dynamic_cluster_same_target_boost", 1.0))
        robots = self.env.robots
        nr = len(robots)
        conflict_w = 0.0
        crowd_sq = 0.0
        for i in range(nr):
            ri = robots[i]
            if ri.collision or ri.success:
                continue
            for j in range(i + 1, nr):
                rj = robots[j]
                if rj.collision or rj.success:
                    continue
                d = cal_distance(ri.px, ri.py, rj.px, rj.py)
                if Dc > 1e-9:
                    crowd_sq += max(0.0, 1.0 - d / Dc) ** 2
                if int(ri.target_id) % K != int(rj.target_id) % K:
                    continue
                if d < d_thr:
                    w = 1.0
                    if cl_boost != 1.0 and (Dc <= 1e-9 or d < Dc):
                        w *= cl_boost
                    conflict_w += w
        if conflict_w > 0.0 and pen_scale != 0.0:
            shared -= pen_scale * conflict_w / n
        crowd_scale = float(getattr(a, "dynamic_crowding_penalty_scale", 0.0))
        c_crowding = 0.0
        if crowd_scale != 0.0 and Dc > 1e-9 and crowd_sq > 0.0:
            c_crowding = crowd_scale * crowd_sq / n
            shared -= c_crowding

        td_prev = getattr(self.env, "dynamic_team_dist_sum_prev", None)
        td_now = float(getattr(self.env, "dynamic_team_dist_sum_this_step", 0.0))
        tp_scale = float(getattr(a, "dynamic_team_dist_progress_scale", 0.15))
        if td_prev is not None and tp_scale != 0.0:
            shared += tp_scale * (float(td_prev) - td_now) / n

        oc_scale = float(getattr(a, "dynamic_target_overcommit_scale", 0.4))
        if use_v2:
            oc_scale *= float(getattr(a, "dynamic_v2_shared_overcommit_mult", 1.0))
        if oc_scale != 0.0:
            counts = {}
            for r in self.env.robots:
                if r.collision is True or r.success is True:
                    continue
                kk = int(r.target_id) % K
                counts[kk] = counts.get(kk, 0) + 1
            excess = sum(max(0, c - 1) for c in counts.values())
            if excess > 0:
                shared -= oc_scale * float(excess) / n

        sw_cost = float(getattr(a, "dynamic_target_switch_penalty", 0.45))
        sw_near_extra_applied = 0.0
        if getattr(robot, "target_switched_this_step", False):
            hold_at = int(getattr(robot, "dynamic_hold_at_switch", 0))
            boost = float(getattr(a, "dynamic_switch_low_hold_boost_scale", 2.5))
            if boost > 0.0:
                sw_cost *= 1.0 + boost / (1.0 + float(max(0, hold_at)))
            relief = float(getattr(a, "dynamic_switch_penalty_relief_when_team_closer", 1.0))
            if (
                td_prev is not None
                and relief < 1.0
                and td_now < float(td_prev) - 1e-5
            ):
                sw_cost *= relief
            extra_sw = float(getattr(a, "dynamic_switch_near_goal_extra", 0.0))
            margin = float(getattr(a, "dynamic_switch_near_goal_margin_radius", 1.75))
            if extra_sw != 0.0:
                thr_g = float(robot.radius) * max(margin, 1e-6)
                for gx, gy in self.env.goal_positions:
                    if cal_distance(robot.px, robot.py, gx, gy) <= thr_g + 1e-9:
                        sw_cost += extra_sw
                        sw_near_extra_applied = extra_sw
                        break
            if in_swap:
                swm = float(getattr(a, "dynamic_reciprocal_swap_switch_penalty_mult", 1.0))
                sw_cost *= swm
            if use_v2:
                if n_nb <= int(getattr(a, "dynamic_switch_sparse_neighbor_max", 1)):
                    sw_cost += float(getattr(a, "dynamic_switch_sparse_extra", 0.9))
                ls = int(getattr(robot, "dynamic_loiter_at_switch", 0))
                if ls >= int(getattr(a, "dynamic_loiter_steps_for_switch_penalty", 6)):
                    sw_cost += float(getattr(a, "dynamic_switch_after_loiter_extra", 1.25))
            ep_c = float(getattr(a, "dynamic_switch_episode_prior_coef", 0.0))
            if ep_c > 0.0:
                sw_cost += ep_c * float(
                    max(0, int(getattr(robot, "dynamic_episode_target_switch_prior", 0)))
                )
            if sw_cost != 0.0:
                shared -= sw_cost

        if self.env.dynamic_formation_success_once:
            shared += float(getattr(a, "dynamic_formation_success_bonus", 50.0)) / n

        if use_v2 and int(getattr(a, "dynamic_formation_time_invariant", 1)) != 0:
            w_form = 1.0
        else:
            w_form = self._formation_time_weight()
        df = float(getattr(a, "dynamic_discount_formation", 0.0))
        ff = max(float(for_feature), 0.0)
        r_formation = -np.sqrt(ff)
        c_formation = df * w_form * r_formation

        r_avoid = 0.0
        if robot.collision == True:
            r_avoid = float(getattr(a, "dynamic_v2_collision_hard_penalty", -95.0)) if use_v2 else -60.0
        else:
            if robot.dmin < robot.discomfort_dist * 2:
                dm = float(getattr(a, "dynamic_v2_dmin_avoid_mult", 1.5)) if use_v2 else 1.0
                r_avoid = -dm * np.exp(-robot.dmin / 3)

        switched = bool(getattr(robot, "target_switched_this_step", False))
        if switched and int(getattr(a, "dynamic_zero_nav_on_target_switch", 1)) != 0:
            nav_scale = 0.0
        elif switched:
            nav_scale = float(getattr(a, "dynamic_nav_scale_on_target_switch", 0.0))
        else:
            nav_scale = 1.0

        r_nav = 0.0
        if robot.pre_dist2goal is not None:
            r_nav = (robot.pre_dist2goal - robot.dist2goal) * 5 * nav_scale
        if r_nav > 0 and robot.collision:
            r_nav = 0.0

        if switched:
            prox_sw = float(getattr(a, "dynamic_prox_scale_on_target_switch", 0.2))
        else:
            prox_sw = 1.0
        r_prox = 0.0
        pr = float(getattr(a, "dynamic_proximity_reward_scale", 0.0))
        if pr != 0.0 and robot.collision is not True and robot.success is not True:
            sig = max(float(getattr(a, "dynamic_proximity_sigma", 8.0)), 1e-6)
            r_prox = pr * float(np.exp(-float(robot.dist2goal) / sig)) * prox_sw

        urg_base = self._dynamic_sparse_urgency_multiplier(n_nb)
        if int(getattr(a, "dynamic_sparse_urgency_only_when_uncontested", 1)) != 0 and contested:
            urg = 1.0
        else:
            urg = urg_base
        r_nav *= urg
        r_prox *= urg
        if use_v2 and cache is not None and cache["role"].get(ridx) == "rest":
            rm = float(getattr(a, "dynamic_cluster_rest_nav_prox_mult", 1.2))
            r_nav *= rm
            r_prox *= rm

        r_arrive = 0.0
        if getattr(robot, "dynamic_just_arrived", False):
            r_arrive = float(getattr(a, "dynamic_arrival_reward", 25.0))

        if use_v2:
            explore_z = {
                "r_explore_undervisible_raw": 0.0,
                "dynamic_explore_shortfall": 0.0,
                "dynamic_explore_n_in_view": 0.0,
                "dynamic_explore_align": 0.0,
                "dynamic_explore_density_mult": 0.0,
                "dynamic_explore_commit_suppressed": 0.0,
            }
            r_explore = 0.0
        else:
            r_explore, explore_z = self._dynamic_explore_undervisible_bonus(
                robot, density_mult=explore_dm
            )
        explore_z = {**explore_z, "dynamic_local_neighbors": float(n_nb)}

        if in_swap and not (
            use_v2 and int(getattr(a, "dynamic_v2_disable_reciprocal_swap_bonus", 1)) != 0
        ):
            r_swap = float(getattr(a, "dynamic_reciprocal_swap_reward_scale", 0.0))
        else:
            r_swap = 0.0

        pcs = float(getattr(a, "dynamic_claimed_target_penalty_scale", 0.0))
        r_claimed = -pcs if (pcs > 0.0 and chasing_claimed) else 0.0

        if use_v2:
            flee_z = {
                "r_low_density_explore_raw": 0.0,
                "dynamic_low_density_mode": 0.0,
                "dynamic_low_density_commit_suppressed": 0.0,
            }
            r_flee = 0.0
        else:
            low_density_active = contested or chasing_claimed
            r_flee, flee_z = self._dynamic_low_density_explore_bonus(
                robot,
                ridx,
                active=low_density_active,
                n_nb=n_nb,
                chasing_claimed=chasing_claimed,
            )

        r_ctde_remain = 0.0
        if not use_v2:
            s_ctde = float(getattr(a, "dynamic_ctde_remaining_target_shaping_scale", 0.0))
            req_cv = int(getattr(a, "dynamic_ctde_remaining_shaping_require_centralized_v", 1)) != 0
            use_cv = bool(getattr(a, "use_centralized_V", False))
            if (
                s_ctde != 0.0
                and (not req_cv or use_cv)
                and robot.collision is not True
                and robot.success is not True
            ):
                p0 = getattr(robot, "pre_dist_nearest_unclaimed", None)
                d0 = getattr(robot, "dist_nearest_unclaimed", None)
                if p0 is not None and d0 is not None:
                    coef = float(getattr(a, "dynamic_ctde_remaining_target_progress_coef", 5.0))
                    r_ctde_remain = s_ctde * coef * (float(p0) - float(d0))
                    if self._dynamic_in_assigned_goal_commit_zone(robot):
                        r_ctde_remain = 0.0

        r_cluster = 0.0
        r_disp = 0.0
        role_s = "none"
        if use_v2 and cache is not None and robot.collision is not True and robot.success is not True:
            role_s = str(cache["role"].get(ridx, "rest"))
            prog_c = float(getattr(a, "dynamic_cluster_progress_coef", 5.0))
            sl = float(getattr(a, "dynamic_cluster_local_unc_shaping_scale", 0.0))
            se = float(getattr(a, "dynamic_cluster_explore_shaping_scale", 0.0))
            pe = float(getattr(a, "dynamic_cluster_explorer_no_target_penalty", 0.0))
            puc = getattr(robot, "pre_dist_unc_near_cluster", None)
            duc = getattr(robot, "dist_unc_near_cluster", None)
            pn = getattr(robot, "pre_dist_nearest_unclaimed", None)
            dn = getattr(robot, "dist_nearest_unclaimed", None)
            if role_s == "local_unc":
                if sl > 0.0 and puc is not None and duc is not None:
                    r_cluster = sl * prog_c * (float(puc) - float(duc))
            elif role_s == "explore":
                if not self._dynamic_in_assigned_goal_commit_zone(robot):
                    po = getattr(robot, "pre_dist_unclaimed_outside_near", None)
                    do = getattr(robot, "dist_unclaimed_outside_near", None)
                    if se > 0.0 and po is not None and do is not None:
                        r_cluster = se * prog_c * (float(po) - float(do))
                    elif se > 0.0 and pn is not None and dn is not None:
                        r_cluster = se * prog_c * (float(pn) - float(dn))
                    elif pe > 0.0:
                        r_cluster = -pe
            elif role_s == "rest":
                beta = float(getattr(a, "dynamic_cluster_rest_dispersion_scale", 0.0))
                sig = max(float(getattr(a, "dynamic_cluster_rest_dispersion_sigma", 4.0)), 1e-6)
                cid = int(cache["cluster_of"].get(ridx, -1))
                if beta > 0.0 and cid >= 0:
                    ssum = 0.0
                    for j in cache["members"].get(cid, []):
                        if j == ridx:
                            continue
                        rj = robots[j]
                        if rj.collision or rj.success:
                            continue
                        d_ij = cal_distance(robot.px, robot.py, rj.px, rj.py)
                        ssum += math.exp(-d_ij / sig)
                    r_disp = -beta * ssum

        discount_avoid = float(getattr(a, "nd_discount_avoid", 50.0))
        discount_nav = float(getattr(a, "nd_discount_nav", 20.0))
        c_avoid = discount_avoid * r_avoid
        c_nav = discount_nav * r_nav
        c_prox = discount_nav * r_prox
        c_explore = discount_nav * r_explore
        c_swap = discount_nav * r_swap
        c_claimed = discount_nav * r_claimed
        c_flee = discount_nav * r_flee
        c_ctde_remain = discount_nav * r_ctde_remain
        c_cluster = discount_nav * (r_cluster + r_disp)
        reward = (
            shared
            + c_formation
            + c_avoid
            + c_nav
            + c_prox
            + r_arrive
            + c_explore
            + c_swap
            + c_claimed
            + c_flee
            + c_ctde_remain
            + c_cluster
        )
        _dn_u = getattr(robot, "dist_nearest_unclaimed", None)
        _pd_nu = getattr(robot, "pre_dist_nearest_unclaimed", None)
        _commit_zone = self._dynamic_in_assigned_goal_commit_zone(robot)
        robot._reward_terms = {
            "reward_mode": "dynamic",
            "shared_team": float(shared),
            "dynamic_team_dist_prev": float(td_prev) if td_prev is not None else float("nan"),
            "dynamic_team_dist_now": float(td_now),
            "dynamic_team_dist_delta": float(td_prev - td_now) if td_prev is not None else float("nan"),
            "dynamic_conflict_weighted": float(conflict_w),
            "dynamic_crowding_sq": float(crowd_sq),
            "c_dynamic_crowding": float(c_crowding),
            "dynamic_switch_near_goal_extra": float(sw_near_extra_applied),
            "dynamic_nav_scale": float(nav_scale),
            "r_formation_raw": float(r_formation),
            "r_avoid_raw": float(r_avoid),
            "r_nav_raw": float(r_nav),
            "r_prox_raw": float(r_prox),
            "r_arrive_raw": float(r_arrive),
            "r_explore_undervisible_raw": float(explore_z["r_explore_undervisible_raw"]),
            "dynamic_explore_shortfall": float(explore_z["dynamic_explore_shortfall"]),
            "dynamic_explore_n_in_view": float(explore_z["dynamic_explore_n_in_view"]),
            "dynamic_explore_align": float(explore_z["dynamic_explore_align"]),
            "dynamic_explore_density_mult": float(explore_z.get("dynamic_explore_density_mult", 1.0)),
            "dynamic_explore_commit_suppressed": float(
                explore_z.get("dynamic_explore_commit_suppressed", 0.0)
            ),
            "dynamic_in_commit_zone": float(1.0 if _commit_zone else 0.0),
            "dynamic_local_neighbors": float(explore_z.get("dynamic_local_neighbors", 0.0)),
            "dynamic_sparse_urgency_mult": float(urg),
            "dynamic_same_target_contested": float(1.0 if contested else 0.0),
            "dynamic_chasing_claimed": float(1.0 if chasing_claimed else 0.0),
            "r_claimed_target_raw": float(r_claimed),
            "c_claimed_target": float(c_claimed),
            "r_low_density_explore_raw": float(flee_z["r_low_density_explore_raw"]),
            "dynamic_low_density_mode": float(flee_z["dynamic_low_density_mode"]),
            "dynamic_low_density_commit_suppressed": float(
                flee_z.get("dynamic_low_density_commit_suppressed", 0.0)
            ),
            "c_low_density_explore": float(c_flee),
            "dist_nearest_unclaimed": float(_dn_u) if _dn_u is not None else float("nan"),
            "pre_dist_nearest_unclaimed": float(_pd_nu) if _pd_nu is not None else float("nan"),
            "r_ctde_remaining_raw": float(r_ctde_remain),
            "c_ctde_remaining_target": float(c_ctde_remain),
            "dynamic_reward_v2": float(1.0 if use_v2 else 0.0),
            "dynamic_cluster_role": role_s,
            "r_cluster_shaping_raw": float(r_cluster),
            "r_cluster_dispersion_raw": float(r_disp),
            "c_cluster_shaping": float(c_cluster),
            "c_formation": float(c_formation),
            "formation_time_w": float(w_form),
            "c_avoid": float(c_avoid),
            "c_nav": float(c_nav),
            "c_prox": float(c_prox),
            "c_explore_undervisible": float(c_explore),
            "dynamic_in_reciprocal_swap": float(1.0 if in_swap else 0.0),
            "r_reciprocal_swap_raw": float(r_swap),
            "c_reciprocal_swap": float(c_swap),
            "dynamic_goal_contention_excess": float(cont_excess),
            "c_goal_contention": float(cont_pen),
            "reward_final": float(reward),
        }
        return np.array([reward])

    def compute_pattern_speed_direction_reward(
        self,
        robot: Any,
        for_feature: float,
        ctx: Optional[RewardContext] = None,
    ) -> np.ndarray:
        """
        TODO: Implement reward shaping based on:
          - agent speed (robot.v)
          - agent direction (robot.theta)
          - target pattern name / template (ctx.pattern_name or env.pattern_name)
          - iteration step (ctx.iteration_step or env.global_time / step index)

        For now, this method is intentionally not used.
        """
        raise NotImplementedError(
            "compute_pattern_speed_direction_reward() is a placeholder for future work."
        )





# reward not modified

# import numpy as np

# from dataclasses import dataclass
# from typing import Optional, Any

# from envs.utils.utils import reach_goal


# @dataclass(frozen=True)
# class RewardContext:
#     """
#     Reward computation context for future extensibility.

#     Note: current default reward implementation does not require most fields.
#     """

#     pattern_name: Optional[str] = None
#     iteration_step: Optional[int] = None
#     # You can extend this with more signals later.
#     extra: Optional[Any] = None


# class RewardCalculator:
#     """
#     Central place to compute rewards.

#     - `compute_default_reward` preserves current EnvCore logic exactly.
#     - `compute_pattern_speed_direction_reward` is a placeholder for future reward shaping.
#     """

#     def __init__(self, env_core: Any):
#         self.env = env_core

#     def compute_default_reward(self, robot: Any, for_feature: float) -> np.ndarray:
#         """
#         Preserve EnvCore.get_reward() current logic.
#         """
#         r_avoid = 0
#         r_goal = 0
#         r_nav = 0
#         r_bonus = 0
#         robot.goal_flag = False
#         # collision
#         if robot.collision == True:
#             r_avoid = -60
#         else:
#             if robot.dmin < robot.discomfort_dist * 2:
#                 r_avoid = -np.exp(-robot.dmin / 3)

#         # formation
#         r_formation = -np.sqrt(for_feature)

#         # bonus (currently not used in final reward)
#         if abs(robot.pre_theta - robot.theta) > 0.7:
#             r_bonus = -1

#         # navigation
#         r_nav += (robot.pre_dist2goal - robot.dist2goal) * 5

#         if r_nav > 0 and robot.collision:
#             r_nav = 0

#         if reach_goal(robot):
#             robot.goal_flag = True
#             if robot.v != 0:
#                 r_goal += 5

#         if self.env.collision_flag:
#             r_goal = 0

#         discount_formation = 0
#         discount_avoid = 50
#         discount_nav = 20
#         discount_goal = 200
#         discount_bonus = 2

#         reward = (
#             discount_formation * r_formation
#             + discount_avoid * r_avoid
#             + discount_nav * r_nav
#             + discount_goal * r_goal  # + discount_bonus * r_bonus
#         )
#         if robot.goal_flag:
#             reward = 0

#         return np.array([reward])

#     def compute_pattern_speed_direction_reward(
#         self,
#         robot: Any,
#         for_feature: float,
#         ctx: Optional[RewardContext] = None,
#     ) -> np.ndarray:
#         """
#         TODO: Implement reward shaping based on:
#           - agent speed (robot.v)
#           - agent direction (robot.theta)
#           - target pattern name / template (ctx.pattern_name or env.pattern_name)
#           - iteration step (ctx.iteration_step or env.global_time / step index)

#         For now, this method is intentionally not used.
#         """
#         raise NotImplementedError(
#             "compute_pattern_speed_direction_reward() is a placeholder for future work."
#         )
