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
        """Linear τ∈[0,1] over episode; multiply formation base weights.

        Uses time at the start of the current transition (global_time was already
        advanced by time_step before reward), so the first step uses τ=0 → start weight.
        """
        a = self.env.args
        t0 = float(getattr(a, "formation_time_weight_start", 1.0))
        t1 = float(getattr(a, "formation_time_weight_end", 1.0))
        tl = max(float(getattr(self.env, "time_limit", 1.0)), 1e-9)
        dt = float(getattr(self.env, "time_step", 0.1))
        gt = float(getattr(self.env, "global_time", 0.0))
        t_elapsed = max(0.0, gt - dt)
        tau = min(1.0, t_elapsed / tl)
        return t0 + (t1 - t0) * tau

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

    def _dynamic_local_active_neighbor_count(self, ridx: int) -> int:
        if ridx < 0:
            return 0
        env = self.env
        a = env.args
        R = float(getattr(a, "dynamic_local_density_radius", 0.0))
        if R <= 1e-9:
            R = float(getattr(a, "dynamic_crowding_dist", 0.0))
        if R <= 1e-9:
            R = 3.0
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

    def compute_default_reward(self, robot: Any, for_feature: float) -> np.ndarray:
        """
        Preserve EnvCore.get_reward() current logic.
        """
        if getattr(self.env, "dynamic_goal_assignment", False):
            return self._compute_dynamic_goal_reward(robot, for_feature)
        a = self.env.args
        r_avoid = 0
        r_goal = 0
        r_nav = 0
        r_bonus = 0
        robot.goal_flag = False
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
        if robot.pre_dist2goal is not None:
            r_nav += (robot.pre_dist2goal - robot.dist2goal) * prog_coef

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

        if reach_goal(robot):
            robot.goal_flag = True
            r_goal += float(getattr(a, "nd_arrival_reward", 0.0))
            if robot.v != 0:
                r_goal += 5

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
            if tr != 0.0:
                entered = bool(
                    robot.pre_dist2goal is not None
                    and robot.pre_dist2goal > robot.radius + 1e-9
                )
                if entered:
                    reward = reward_mid + tr
                    nd_tr_applied = tr
                else:
                    reward = 0.0
            else:
                reward = 0.0

        robot._reward_terms = {
            "reward_mode": "static_nd",
            "r_avoid_raw": float(r_avoid),
            "r_formation_raw": float(r_formation),
            "r_nav_raw": float(r_nav),
            "r_goal_raw": float(r_goal),
            "r_bonus_raw": float(r_bonus),
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
            **{k: float(v) for k, v in comm_terms.items()},
        }

        return np.array([reward])



    def _compute_dynamic_goal_reward(self, robot: Any, for_feature: float) -> np.ndarray:
        """Team travel, conflicts, coordinated distance + assignment shaping, arrival signal.

        Anti-exploit: same-target penalty is stronger when agents are crowded; optional arrival
        gate (env) requires entering from outside the goal disk; target switches near goals cost
        extra; r_nav is suppressed on target switches (gx,gy discontinuity).

        Local density: many nearby teammates → stronger explore outside R_vis; few neighbors →
        higher nav/prox urgency toward current target; choosing another agent's claimed goal → penalty.
        """
        a = self.env.args
        n = max(1, self.env.robot_num)
        try:
            ridx = self.env.robots.index(robot)
        except ValueError:
            ridx = -1
        n_nb = self._dynamic_local_active_neighbor_count(ridx)
        explore_dm = self._dynamic_explore_density_multiplier(n_nb)

        shared = 0.0
        shared -= float(getattr(a, "dynamic_path_reward_scale", 1.0)) * self.env.dynamic_step_travel_sum / n

        swap_idx = self._dynamic_find_reciprocal_swap_indices()
        cont_pen, cont_excess = self._dynamic_goal_contention_penalty()
        shared -= cont_pen

        in_swap = ridx >= 0 and ridx in swap_idx

        d_thr = float(getattr(a, "dynamic_same_target_conflict_dist", 2.0))
        pen_scale = float(getattr(a, "dynamic_same_target_penalty_scale", 2.0))
        Dc = float(getattr(a, "dynamic_crowding_dist", 0.0))
        cl_boost = float(getattr(a, "dynamic_cluster_same_target_boost", 1.0))
        K = int(self.env.num_goal_targets)
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
        if oc_scale != 0.0:
            K = int(self.env.num_goal_targets)
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
            if sw_cost != 0.0:
                shared -= sw_cost

        if self.env.dynamic_formation_success_once:
            shared += float(getattr(a, "dynamic_formation_success_bonus", 50.0)) / n

        w_form = self._formation_time_weight()
        df = float(getattr(a, "dynamic_discount_formation", 0.0))
        ff = max(float(for_feature), 0.0)
        r_formation = -np.sqrt(ff)
        c_formation = df * w_form * r_formation

        r_avoid = 0.0
        if robot.collision == True:
            r_avoid = -60.0
        else:
            if robot.dmin < robot.discomfort_dist * 2:
                r_avoid = -np.exp(-robot.dmin / 3)

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

        urg = self._dynamic_sparse_urgency_multiplier(n_nb)
        r_nav *= urg
        r_prox *= urg

        r_arrive = 0.0
        if getattr(robot, "dynamic_just_arrived", False):
            r_arrive = float(getattr(a, "dynamic_arrival_reward", 25.0))

        r_explore, explore_z = self._dynamic_explore_undervisible_bonus(
            robot, density_mult=explore_dm
        )
        explore_z = {**explore_z, "dynamic_local_neighbors": float(n_nb)}

        r_swap = float(getattr(a, "dynamic_reciprocal_swap_reward_scale", 0.0)) if in_swap else 0.0

        r_claimed = 0.0
        pcs = float(getattr(a, "dynamic_claimed_target_penalty_scale", 0.0))
        if (
            pcs > 0.0
            and ridx >= 0
            and robot.collision is not True
            and robot.success is not True
        ):
            cb = getattr(self.env, "claimed_by", None)
            if cb is not None and len(cb) == K:
                tk = int(robot.target_id) % K
                owner = int(cb[tk])
                if owner >= 0 and owner != ridx:
                    r_claimed = -pcs

        discount_avoid = float(getattr(a, "nd_discount_avoid", 50.0))
        discount_nav = float(getattr(a, "nd_discount_nav", 20.0))
        c_avoid = discount_avoid * r_avoid
        c_nav = discount_nav * r_nav
        c_prox = discount_nav * r_prox
        c_explore = discount_nav * r_explore
        c_swap = discount_nav * r_swap
        c_claimed = discount_nav * r_claimed
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
        )
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
            "dynamic_local_neighbors": float(explore_z.get("dynamic_local_neighbors", 0.0)),
            "dynamic_sparse_urgency_mult": float(urg),
            "r_claimed_target_raw": float(r_claimed),
            "c_claimed_target": float(c_claimed),
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
