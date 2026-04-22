import logging
import math
import numpy as np
import rvo2
from numpy.linalg import norm
from envs.utils.human import Human
from envs.utils.info import *
from envs.utils.utils import cal_distance,reach_goal,get_weight
from envs.utils.state_lux import JointState
from envs.utils.robot import Robot
import time
from envs.utils.reward_calculator import RewardCalculator
from envs.utils.hungarian_opt import assignment_cost_for_target_ids, optimal_assignment_cost


def _parse_csv_ints_arg(s):
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_names_arg(s):
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def select_font_pattern_targets_for_args(args, *, log_selection=True):
    """
    Select font pattern (formation template) the same way as EnvCore.
    Used by render to size num_agents before env construction.
    Returns (pattern_name, pattern_coords).
    """
    from dataset.FontPatternLoader import FontPatternLoader

    use_render = getattr(args, "use_render", False)
    if use_render:
        lengths = _parse_csv_ints_arg(getattr(args, "render_font_pattern_lengths", "10"))
        policy = getattr(args, "render_font_pattern_policy", "all")
        names = _parse_csv_names_arg(getattr(args, "render_font_pattern_names", ""))
        allow_repeat = True
    else:
        lengths = [int(getattr(args, "train_font_pattern_length", 10))]
        policy = getattr(args, "train_font_pattern_policy", "all")
        names = _parse_csv_names_arg(getattr(args, "train_font_pattern_names", ""))
        allow_repeat = bool(getattr(args, "train_font_pattern_allow_repeat", False))

    env_rank = int(getattr(args, "env_rank", 0))
    threads = int(getattr(args, "n_rollout_threads", 1))

    loader = FontPatternLoader(
        target_lengths=lengths,
        dataset_dir=getattr(args, "font_pattern_dataset_dir", "./dataset"),
    )

    pool_all = []
    for L in lengths:
        pool_all.extend(loader.data_store.get(L, []))

    if not pool_all:
        raise RuntimeError(f"No font patterns loaded for lengths={lengths}.")

    if policy == "only":
        if not names:
            raise RuntimeError("train_font_pattern_policy/render_font_pattern_policy is 'only' but no names provided.")
        pool_allowed = [p for p in pool_all if p.get("name") in set(names)]
    else:
        pool_allowed = list(pool_all)

    if not pool_allowed:
        raise RuntimeError(f"Font pattern policy '{policy}' produced empty pool. names={names}, lengths={lengths}")

    ordered_pool = []
    used_names = set()
    if policy == "must_contain":
        if not names:
            raise RuntimeError("Font pattern policy 'must_contain' requires non-empty pattern names.")

        required_set = set(names)
        found_set = {p.get("name") for p in pool_allowed}
        missing = required_set - found_set
        if missing:
            raise RuntimeError(f"Missing required font patterns: {sorted(list(missing))}")

        if len(names) > threads:
            raise RuntimeError(
                f"Not enough env threads (threads={threads}) to include all required patterns (required={names})."
            )

        for n in names:
            for p in pool_allowed:
                if p.get("name") == n:
                    ordered_pool.append(p)
                    used_names.add(n)
                    break

    for p in pool_allowed:
        n = p.get("name")
        if n not in used_names:
            ordered_pool.append(p)

    pattern_count = len(ordered_pool)
    if threads > pattern_count:
        allow_repeat = True

    if not allow_repeat and threads > pattern_count:
        raise RuntimeError("Repetition disabled but env threads exceed available patterns.")

    if allow_repeat:
        idx = env_rank % pattern_count
    else:
        idx = env_rank

    chosen = ordered_pool[idx]
    pattern_name = chosen.get("name", "unknown")
    pattern_coords = chosen.get("coordinates", [])

    if not pattern_coords:
        raise RuntimeError(f"Chosen font pattern '{pattern_name}' has empty coordinates.")

    if log_selection:
        if not use_render:
            print(f"[font-pattern][train] env_rank={env_rank}/{threads-1}, length={lengths[0]}, name={pattern_name}")
        else:
            print(f"[font-pattern][render] env_rank={env_rank}, length(s)={lengths}, name={pattern_name}")

    return pattern_name, pattern_coords


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self, args):
        self.args = args
        self.time_step = args.time_step
        self.time_limit = args.episode_length * self.time_step
        self.method = args.method
        self.humans = None
        self.robots = None
        self.global_time = None
        self.human_times = None
        self.robot_num = args.num_agents
        self.human_num = args.num_humans
        self.att_agents = args.num_attention_agents
        self.robot_obs_dim = args.robot_obs_dim
        self.human_obs_dim = args.human_obs_dim
        self.obs_dim = max(self.robot_obs_dim, self.human_obs_dim)
        self.vel_action_dim = args.vel_action_dim
        self.dir_action_dim = args.dir_action_dim
        self.discomfort_dist = args.dcf_dist
        self.base_v = args.base_v
        # simulation configuration
        self.config = None
        self.randomize_attributes = args.randomize_attributes
        self.train_val_sim = args.human_action
        self.square_width = args.square_width
        self.circle_radius = args.circle_radius
        # for visualization
        self.total_obs = None

        # for formulate
        self.observation_states = None
        self.attention_weights = None
        self.L_des = None
        self.laplacian_S_L = float("nan")
        self.laplacian_S_L_prev = float("nan")
        self.for_edge = args.for_edge

        self.is_reset = None
        self.collision_flag = None

        # centralized reward computation object (default implementation preserves logic)
        self.reward_calculator = RewardCalculator(self)
        
        # print('human number:',self.human_num)

        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info('simulation: {}'.format(self.train_val_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        # font pattern selection (used to set robot goals/formation templates)
        self.pattern_name, self.s_shape_targets = self._select_font_pattern_targets()

        # K == swarm size here; reset() may set len(goal_positions) when dynamic goals are built
        self.num_goal_targets = self.robot_num

        self.undetermined_goal_assignment = bool(getattr(args, "enable_undetermined_goal", False))
        self.undetermined_goal_v3 = bool(getattr(args, "enable_undetermined_goal_v3", False))
        self.undetermined_goal_v2 = bool(getattr(args, "enable_undetermined_goal_v2", False))
        self.undetermined_v2_goal_slots = max(1, int(getattr(args, "undetermined_v2_goal_slots", 10)))
        self.undetermined_v3_comm_P = max(0, int(getattr(args, "undetermined_v3_comm_ally_slots", 6)))
        self.undetermined_v3_comm_H = max(0, int(getattr(args, "undetermined_v3_comm_human_slots", 4)))
        if self.undetermined_goal_v3:
            if not self.undetermined_goal_assignment:
                raise ValueError("enable_undetermined_goal_v3 requires enable_undetermined_goal")
            if not bool(getattr(args, "use_attn_comm_actor", False)):
                raise ValueError("enable_undetermined_goal_v3 requires use_attn_comm_actor (integrated communication)")
            if str(getattr(args, "architecture_mode", "default")) != "attn_undetermined_goal":
                raise ValueError(
                    "enable_undetermined_goal_v3 requires --architecture_mode attn_undetermined_goal"
                )
            if self.undetermined_goal_v2:
                raise ValueError("enable_undetermined_goal_v3 is incompatible with enable_undetermined_goal_v2")
            if str(getattr(args, "undet_v2_head_arch", "dot_product")) == "pair_mlp":
                raise ValueError("enable_undetermined_goal_v3 requires --undet_v2_head_arch dot_product (hybrid obs)")
        if self.undetermined_goal_v2 and not self.undetermined_goal_assignment:
            raise ValueError("enable_undetermined_goal_v2 requires enable_undetermined_goal")
        self.undetermined_v2_exchange = bool(getattr(args, "enable_undetermined_v2_exchange", False))
        if self.undetermined_v2_exchange:
            if not self.undetermined_goal_v2:
                raise ValueError("enable_undetermined_v2_exchange requires enable_undetermined_goal_v2")
            if self.undetermined_goal_v3:
                raise ValueError("enable_undetermined_v2_exchange is not supported with enable_undetermined_goal_v3")
        _ex_r = getattr(args, "undetermined_v2_exchange_radius", None)
        self.undetermined_v2_exchange_radius = float(
            _ex_r if _ex_r is not None else float(getattr(args, "undetermined_comm_radius", 6.0))
        )
        # v2_exchange reward shaping: fleet bottleneck (max dist2goal) & sum dist; swap-step gains (see RewardCalculator)
        self.undetermined_v2_fleet_M_prev = float("nan")
        self.undetermined_v2_fleet_S_prev = float("nan")
        self.undetermined_v2_fleet_M_drop = 0.0
        self.undetermined_v2_fleet_S_drop = 0.0
        self.undetermined_v2_exchange_step_M_gain = 0.0
        self.undetermined_v2_exchange_step_S_gain = 0.0
        self.undetermined_v2_exchange_agent_mask = np.zeros(self.robot_num, dtype=np.bool_)
        self.dynamic_goal_assignment = bool(getattr(args, "enable_dynamic_goal_assignment", False))
        if self.undetermined_goal_assignment:
            self.dynamic_goal_assignment = False
        self.undetermined_obs_goal_radius = float(getattr(args, "undetermined_obs_goal_radius", 5.0))
        self.undetermined_comm_radius = float(getattr(args, "undetermined_comm_radius", 6.0))
        self.dynamic_obs_pack_version = str(
            getattr(args, "dynamic_obs_pack_version", "slots")
        )
        self.dynamic_slot_m = 1
        self.dynamic_vis_radius = float(getattr(args, "dynamic_target_vis_radius", 5.0))
        if self.dynamic_goal_assignment:
            if self.dynamic_obs_pack_version == "legacy":
                self.dynamic_slot_m = self.num_goal_targets
            else:
                sc = int(getattr(args, "dynamic_target_slot_count", 10))
                self.dynamic_slot_m = max(1, min(sc, self.num_goal_targets))
        self.use_neighbor_attn_actor = (
            bool(self.dynamic_goal_assignment)
            and self.dynamic_obs_pack_version != "legacy"
            and bool(getattr(args, "use_neighbor_attn_lstm_actor", False))
        )
        self.actor_neighbor_p = 0
        if self.use_neighbor_attn_actor:
            self.actor_neighbor_p = min(
                int(getattr(args, "actor_neighbor_n", 10)),
                max(0, self.robot_num - 1),
            )
        self.neighbor_actor_feat_dim = 7
        self.use_attn_comm_actor = bool(getattr(args, "use_attn_comm_actor", False)) and (
            not self.dynamic_goal_assignment
        )
        _nn_def = int(getattr(args, "neighbor_n", 10))
        _as = getattr(args, "attn_comm_ally_slots", None)
        _hs = getattr(args, "attn_comm_human_slots", None)
        self.attn_comm_P = max(0, int(_as if _as is not None else _nn_def))
        self.attn_comm_H = max(0, int(_hs if _hs is not None else _nn_def))
        _ar = getattr(args, "attn_comm_radius", None)
        self.attn_comm_radius = float(
            _ar if _ar is not None else float(getattr(args, "neighbor_radius", 5.0))
        )
        self.attn_comm_msg_dim = max(8, int(getattr(args, "attn_comm_message_dim", 16)))
        self.agent_broadcast_msg = np.zeros((self.robot_num, self.attn_comm_msg_dim), dtype=np.float32)
        self.agent_broadcast_msg_prev = np.zeros((self.robot_num, self.attn_comm_msg_dim), dtype=np.float32)
        self.goal_positions = None
        self.claimed_by = None
        self.undetermined_hungarian_bonus = np.zeros(self.robot_num, dtype=np.float64)
        self.goal_centroid_xy = (0.0, 0.0)
        # Cons-DecAF paper reward lag state (arXiv:2307.12287 Eq.(1)-(2))
        self._cons_decaf_rf_prev = 0.0
        self._cons_decaf_rv_prev = 0.0
        self.dynamic_step_travel_sum = 0.0
        self.dynamic_same_target_conflict_pairs = 0
        self.dynamic_formation_success_once = False
        self.dynamic_episode_had_collision = False
        self.dynamic_team_dist_sum_prev = None
        self.dynamic_team_dist_sum_this_step = 0.0

    def _select_font_pattern_targets(self):
        """Delegate to module-level selector (shared with render.py)."""
        return select_font_pattern_targets_for_args(self.args, log_selection=True)

    def generate_remote_human_position(self):
        for i,human in enumerate(self.humans):
            human.set(10e5+i, 10e5+i, 10e5+i, 10e5+i, 0, 0, 0)

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.args)
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.args)
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in self.robots + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_square_crossing_human(self):
        human = Human(self.args)
        radius = human.radius
        if self.randomize_attributes:
            human.sample_random_attributes()
        sign = np.random.random()
        revise = 0
        while True:
            px = np.random.uniform(-10,10) + revise
            py = np.random.uniform(-10,10) + revise
            collide = False
            for agent in self.robots + self.humans:
                if norm((px - agent.px, py - agent.py)) < (radius + agent.radius + self.discomfort_dist)*2 or \
                    norm((px - agent.gx, py - agent.gy)) < (radius + agent.radius + self.discomfort_dist)*2:
                    collide = True
                    break
            if not collide:
                break
            revise += 1

        revise = 0
        while True:
            if sign >= 0.7: # 0.7
                gx = np.random.uniform(-10,10) - 1 -revise
                gy = np.random.uniform(-10,10) + 1 + revise
            elif 0.3 <= sign < 0.7:   # 0.3~0.7
                gx = px
                gy = py
            else:
                gx = px - self.square_width/2 - revise
                gy = px - self.square_width/2 - revise
            collide = False
            for agent in self.robots + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
            revise += 1
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_circle_crossing_human(self):
        human = Human(self.args)
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            #  check robot collide
            for agent in self.robots + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human
        

    def _sample_collision_free_robot_starts(self):
        args = self.args
        x0, x1 = args.robot_init_x_min, args.robot_init_x_max
        y0, y1 = args.robot_init_y_min, args.robot_init_y_max
        margin = args.robot_init_min_separation_margin
        rr = self.robots[0].radius
        positions = []
        for _i in range(self.robot_num):
            for _ in range(8000):
                px = float(np.random.uniform(x0, x1))
                py = float(np.random.uniform(y0, y1))
                ok = True
                for (qx, qy) in positions:
                    if cal_distance(px, py, qx, qy) < 2 * rr + margin:
                        ok = False
                        break
                if ok:
                    positions.append((px, py))
                    break
            else:
                raise RuntimeError(
                    "Failed to sample collision-free robot starts; widen robot_init_* bounds or reduce num_agents."
                )
        return positions

    def _cluster_comm_reference_radius(self) -> float:
        """Radius used to pack cluster_comm spawns (pairwise geometry vs comm / optional vis radii)."""
        args = self.args
        ovr = getattr(args, "robot_init_cluster_comm_radius", None)
        if ovr is not None and float(ovr) > 0.0:
            return float(ovr)
        mode = str(getattr(args, "robot_init_cluster_radius_mode", "comm"))
        if mode == "comm_vis_adaptive":
            vals: list[float] = []
            if float(self.attn_comm_radius) > 1e-6:
                vals.append(float(self.attn_comm_radius))
            if bool(getattr(self, "undetermined_goal_assignment", False)):
                if float(self.undetermined_comm_radius) > 1e-6:
                    vals.append(float(self.undetermined_comm_radius))
                if float(self.undetermined_obs_goal_radius) > 1e-6:
                    vals.append(float(self.undetermined_obs_goal_radius))
            if bool(getattr(self, "dynamic_goal_assignment", False)):
                if float(self.dynamic_vis_radius) > 1e-6:
                    vals.append(float(self.dynamic_vis_radius))
            if str(getattr(args, "agent_state_mode", "all")) == "nearest_n_radius":
                nr = float(getattr(args, "neighbor_radius", 0.0))
                if nr > 1e-6:
                    vals.append(nr)
            if not vals:
                return 6.0
            return float(min(vals))
        a = float(self.attn_comm_radius)
        u = float(self.undetermined_comm_radius)
        pos = [x for x in (a, u) if x > 1e-6]
        if not pos:
            return 6.0
        if len(pos) == 1:
            return pos[0]
        return float(min(pos))

    def _sample_clustered_robot_starts_in_comm_range(self):
        """
        Place robots on a regular polygon around a random center so they are tightly grouped.
        Ring radius is chosen so adjacent agents respect robot_init_min_separation_margin, and
        (when feasible) the diameter stays within ~cluster comm reference radius.
        """
        args = self.args
        x0, x1 = float(args.robot_init_x_min), float(args.robot_init_x_max)
        y0, y1 = float(args.robot_init_y_min), float(args.robot_init_y_max)
        margin = float(args.robot_init_min_separation_margin)
        rr = float(self.robots[0].radius)
        n_ag = int(self.robot_num)
        r_comm = float(self._cluster_comm_reference_radius())

        if n_ag <= 1:
            for _ in range(8000):
                px = float(np.random.uniform(x0, x1))
                py = float(np.random.uniform(y0, y1))
                return [(px, py)]
            raise RuntimeError("cluster_comm: failed to sample center for single agent.")

        sin_half = math.sin(math.pi / float(n_ag))
        r_min = (2.0 * rr + margin) / (2.0 * max(sin_half, 1e-9))
        r_cap = 0.48 * r_comm
        if r_min <= r_cap:
            r_ring = r_min
        else:
            r_ring = r_min
            if not getattr(self, "_cluster_comm_radius_warned", False):
                logging.warning(
                    "[EnvCore] robot_initial_spawn_mode=cluster_comm: minimum ring radius %.4f exceeds %.4f "
                    "(~half of comm reference %.4f); some pairs may lie outside nominal comm range.",
                    r_min,
                    r_cap,
                    r_comm,
                )
                self._cluster_comm_radius_warned = True

        for _attempt in range(8000):
            cx = float(np.random.uniform(x0, x1))
            cy = float(np.random.uniform(y0, y1))
            phase = float(np.random.uniform(0.0, 2.0 * math.pi))
            positions: list[tuple[float, float]] = []
            ok = True
            for i in range(n_ag):
                ang = phase + (2.0 * math.pi * float(i)) / float(n_ag)
                px = cx + r_ring * math.cos(ang)
                py = cy + r_ring * math.sin(ang)
                if not (x0 <= px <= x1 and y0 <= py <= y1):
                    ok = False
                    break
                positions.append((px, py))
            if not ok:
                continue
            for i in range(n_ag):
                for j in range(i + 1, n_ag):
                    if cal_distance(positions[i][0], positions[i][1], positions[j][0], positions[j][1]) < (
                        2.0 * rr + margin - 1e-9
                    ):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return positions

        raise RuntimeError(
            "cluster_comm: could not place agents inside robot_init_* bounds; widen the box or reduce num_agents."
        )

    def _sample_clustered_robot_starts_in_disk(self):
        """
        Uniform random positions inside a disk (rejection sampling) with pairwise min separation.
        Disk radius is capped by cluster reference radius and init box; center is uniform in the valid box.
        """
        args = self.args
        x0, x1 = float(args.robot_init_x_min), float(args.robot_init_x_max)
        y0, y1 = float(args.robot_init_y_min), float(args.robot_init_y_max)
        margin = float(args.robot_init_min_separation_margin)
        rr = float(self.robots[0].radius)
        n_ag = int(self.robot_num)
        sep = 2.0 * rr + margin
        r_ref = float(self._cluster_comm_reference_radius())
        box_w = x1 - x0
        box_h = y1 - y0
        r_box = 0.5 * max(1e-6, min(box_w, box_h)) - 1e-3
        r_cap = min(0.48 * r_ref, r_box)

        def try_pack(cx: float, cy: float, r_disk: float):
            if r_disk <= 1e-9:
                return None
            if cx - r_disk < x0 or cx + r_disk > x1 or cy - r_disk < y0 or cy + r_disk > y1:
                return None
            pos: list[tuple[float, float]] = []
            for _ in range(n_ag):
                for _t in range(400):
                    ang = float(np.random.uniform(0.0, 2.0 * math.pi))
                    rad = r_disk * math.sqrt(float(np.random.uniform(0.0, 1.0)))
                    px = cx + rad * math.cos(ang)
                    py = cy + rad * math.sin(ang)
                    if not (x0 <= px <= x1 and y0 <= py <= y1):
                        continue
                    ok = True
                    for qx, qy in pos:
                        if cal_distance(px, py, qx, qy) < sep - 1e-9:
                            ok = False
                            break
                    if ok:
                        pos.append((px, py))
                        break
                else:
                    return None
            return pos

        if n_ag <= 1:
            for _ in range(8000):
                px = float(np.random.uniform(x0, x1))
                py = float(np.random.uniform(y0, y1))
                return [(px, py)]
            raise RuntimeError("cluster_disk: failed to sample single-agent position.")

        for scale in range(28):
            r_disk = min(r_cap * (1.06**float(scale)), r_box)
            if r_disk < 0.5 * sep:
                continue
            for _attempt in range(500):
                cx = float(np.random.uniform(x0 + r_disk, x1 - r_disk))
                cy = float(np.random.uniform(y0 + r_disk, y1 - r_disk))
                packed = try_pack(cx, cy, r_disk)
                if packed is not None:
                    return packed

        raise RuntimeError(
            "cluster_disk: could not place agents; widen robot_init_* , reduce num_agents, or raise comm/vis radii."
        )

    def _robot_forward_unit_xy(self, robot) -> tuple[float, float]:
        speed_eps = float(getattr(self.args, "undetermined_v2_exchange_forward_speed_eps", 1e-3))
        vx = float(robot.vx) if robot.vx is not None else 0.0
        vy = float(robot.vy) if robot.vy is not None else 0.0
        sp = math.hypot(vx, vy)
        if sp > speed_eps:
            return vx / sp, vy / sp
        th = float(robot.theta) if robot.theta is not None else 0.0
        return math.cos(th), math.sin(th)

    def _point_in_forward_cone_strict(
        self, px: float, py: float, fx: float, fy: float, qx: float, qy: float, cos_thr: float
    ) -> bool:
        dx = qx - px
        dy = qy - py
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return False
        dx /= dist
        dy /= dist
        return (fx * dx + fy * dy) > cos_thr + 1e-12

    def _fleet_M_S_for_tids(self, tids: list[int]) -> tuple[float, float]:
        K = int(self.num_goal_targets)
        ds: list[float] = []
        for k, r in enumerate(self.robots):
            if r.collision or r.success:
                ds.append(0.0)
                continue
            tid = int(tids[k]) % K
            gx, gy = self.goal_positions[tid]
            ds.append(float(cal_distance(r.px, r.py, gx, gy)))
        if not ds:
            return 0.0, 0.0
        return float(max(ds)), float(sum(ds))

    def _undetermined_v2_exchange_cone_mutual_greedy_m(self) -> None:
        """
        Velocity forward-cone (both agents), strict mutual shorter distance to swapped targets, then
        greedy rounds: among unused agents pick the pair that yields minimum fleet M after a swap
        while strictly lowering M vs current (heuristic to approximate multi-pair min-max).
        """
        K = int(self.num_goal_targets)
        R_ex = float(self.undetermined_v2_exchange_radius)
        max_swaps = max(1, int(getattr(self.args, "undetermined_v2_exchange_max_pairs_per_step", 1)))
        ignore_pending = bool(getattr(self.args, "undetermined_v2_exchange_ignore_pending", False))
        half_deg = float(getattr(self.args, "undetermined_v2_exchange_cone_half_deg", 60.0))
        cos_thr = math.cos(math.radians(half_deg))

        def dist_assigned(idx: int) -> float:
            r = self.robots[idx]
            if r.collision or r.success:
                return 0.0
            tid = int(r.target_id) % K
            gx, gy = self.goal_positions[tid]
            return cal_distance(r.px, r.py, gx, gy)

        cur_dist0 = [dist_assigned(k) for k in range(self.robot_num)]
        M0 = float(max(cur_dist0) if cur_dist0 else 0.0)
        S0 = float(sum(cur_dist0))

        def tid_list() -> list[int]:
            return [int(self.robots[k].target_id) % K for k in range(self.robot_num)]

        def mutual_strict(ri, rj, ti: int, tj: int) -> bool:
            gxi, gyi = self.goal_positions[ti]
            gxj, gyj = self.goal_positions[tj]
            d_i_ti = cal_distance(ri.px, ri.py, gxi, gyi)
            d_j_tj = cal_distance(rj.px, rj.py, gxj, gyj)
            d_i_tj = cal_distance(ri.px, ri.py, gxj, gyj)
            d_j_ti = cal_distance(rj.px, rj.py, gxi, gyi)
            return bool(d_i_tj < d_i_ti and d_j_ti < d_j_tj)

        def cones_ok(ri, rj) -> bool:
            fi_x, fi_y = self._robot_forward_unit_xy(ri)
            fj_x, fj_y = self._robot_forward_unit_xy(rj)
            if not self._point_in_forward_cone_strict(
                float(ri.px), float(ri.py), fi_x, fi_y, float(rj.px), float(rj.py), cos_thr
            ):
                return False
            if not self._point_in_forward_cone_strict(
                float(rj.px), float(rj.py), fj_x, fj_y, float(ri.px), float(ri.py), cos_thr
            ):
                return False
            return True

        agent_used = [False] * self.robot_num
        n_swaps = 0

        while n_swaps < max_swaps:
            tids = tid_list()
            M_cur, _S_cur = self._fleet_M_S_for_tids(tids)
            best_M = None
            best_pair = None
            for i in range(self.robot_num):
                if agent_used[i]:
                    continue
                ri = self.robots[i]
                if ri.collision or ri.success:
                    continue
                if not ignore_pending and bool(getattr(ri, "undetermined_target_pending", False)):
                    continue
                for j in range(i + 1, self.robot_num):
                    if agent_used[j]:
                        continue
                    rj = self.robots[j]
                    if rj.collision or rj.success:
                        continue
                    if not ignore_pending and bool(getattr(rj, "undetermined_target_pending", False)):
                        continue
                    if cal_distance(ri.px, ri.py, rj.px, rj.py) > R_ex + 1e-9:
                        continue
                    ti = int(tids[i]) % K
                    tj = int(tids[j]) % K
                    if ti == tj:
                        continue
                    if not mutual_strict(ri, rj, ti, tj):
                        continue
                    if not cones_ok(ri, rj):
                        continue
                    t2 = list(tids)
                    t2[i], t2[j] = t2[j], t2[i]
                    M_after, _ = self._fleet_M_S_for_tids(t2)
                    if M_after < M_cur - 1e-9:
                        if best_M is None or M_after < best_M - 1e-9:
                            best_M = float(M_after)
                            best_pair = (i, j)
                        elif best_pair is not None and abs(M_after - best_M) <= 1e-9:
                            if (i, j) < best_pair:
                                best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            ri, rj = self.robots[i], self.robots[j]
            ti = int(ri.target_id) % K
            tj = int(rj.target_id) % K
            ri.target_id = tj
            rj.target_id = ti
            ri.undetermined_target_pending = False
            rj.undetermined_target_pending = False
            self.undetermined_v2_exchange_agent_mask[i] = True
            self.undetermined_v2_exchange_agent_mask[j] = True
            agent_used[i] = True
            agent_used[j] = True
            n_swaps += 1

        if n_swaps > 0:
            new_dist = [dist_assigned(k) for k in range(self.robot_num)]
            M_end = float(max(new_dist) if new_dist else 0.0)
            S_end = float(sum(new_dist))
            self.undetermined_v2_exchange_step_M_gain = float(M0 - M_end)
            self.undetermined_v2_exchange_step_S_gain = float(S0 - S_end)
            self._undetermined_sync_all_goals()
            self._undetermined_auction_duplicate_targets()
            self._compute_undetermined_hungarian_shaping()

    def _attn_comm_tail_vector(self, robot_index, robot, for_feature, P: int, H: int) -> np.ndarray:
        """
        Ally geometry (6*P) + last-step recv messages (msg_dim*P) + human feats (5*H) + obstacle summary (4).
        P/H are explicit slot counts (v3 uses fixed P,H decoupled from num_agents; standalone AttnComm uses attn_comm_P/H).
        """
        px, py = float(robot.px), float(robot.py)
        theta = float(robot.theta)
        P = max(0, int(P))
        H = max(0, int(H))
        Rc = max(float(self.attn_comm_radius), 1e-6)
        scale = max(Rc, 1.0)
        others = []
        for j, oj in enumerate(self.robots):
            if j == robot_index:
                continue
            d = float(cal_distance(px, py, oj.px, oj.py))
            if d > Rc + 1e-9:
                continue
            others.append(
                (
                    d,
                    j,
                    float(oj.px - px) / scale,
                    float(oj.py - py) / scale,
                    d / scale,
                    float(oj.v),
                    float(np.cos(float(oj.theta) - theta)),
                    float(np.sin(float(oj.theta) - theta)),
                )
            )
        others.sort(key=lambda t: t[0])
        ally_feats: list[float] = []
        recv_feats: list[float] = []
        for idx in range(P):
            if idx < len(others):
                t = others[idx]
                ally_feats.extend(t[2:8])
                jid = int(t[1])
                recv_feats.extend(self.agent_broadcast_msg[jid].tolist())
            else:
                ally_feats.extend([0.0] * 6)
                recv_feats.extend([0.0] * self.attn_comm_msg_dim)
        hum_list = []
        for human in self.humans:
            d = float(cal_distance(px, py, human.px, human.py))
            if d > Rc + 1e-9:
                continue
            spd = float(np.hypot(float(human.vx), float(human.vy)))
            phi = float(np.arctan2(float(human.py) - py, float(human.px) - px))
            rel_ang = float(np.arctan2(np.sin(phi - theta), np.cos(phi - theta)) / np.pi)
            hum_list.append(
                (
                    d,
                    float(human.px - px) / scale,
                    float(human.py - py) / scale,
                    d / scale,
                    spd / 2.0,
                    rel_ang,
                )
            )
        hum_list.sort(key=lambda t: t[0])
        hum_feats: list[float] = []
        for idx in range(H):
            if idx < len(hum_list):
                hum_feats.extend(hum_list[idx][1:6])
            else:
                hum_feats.extend([0.0] * 5)
        dmin = float(getattr(robot, "dmin", 1e6))
        obst = [
            float(for_feature) / 100.0,
            min(dmin, 30.0) / 30.0,
            float(robot.vx_formation),
            float(robot.vy_formation),
        ]
        tail = np.array(ally_feats + recv_feats + hum_feats + obst, dtype=np.float32)
        _md = int(self.attn_comm_msg_dim)
        _exp = P * 6 + P * _md + H * 5 + 4
        assert tail.shape[0] == _exp, (tail.shape[0], _exp, P, H, _md)
        return tail

    def _pack_robot_obs_row_attn_comm(self, robot_index, robot, for_feature):
        """Fixed targets: self + in-radius allies/humans (nearest-first, capped by slots) + summary."""
        px, py = float(robot.px), float(robot.py)
        P, H = self.attn_comm_P, self.attn_comm_H
        base = [
            robot.gx - px,
            robot.gy - py,
            robot.v,
            robot.theta,
            for_feature,
            robot.vx_formation,
            robot.vy_formation,
        ]
        tail = self._attn_comm_tail_vector(robot_index, robot, for_feature, P, H)
        vec = np.array(base + tail.tolist() + [px, py], dtype=np.float32)
        # Standalone AttnComm row only; hybrid mode sets self.robot_obs_dim to v2+tail (see compute_undetermined_v2_attn_hybrid_robot_obs_dim).
        _p, _h, _md = int(P), int(H), int(self.attn_comm_msg_dim)
        _attn_row_len = 7 + _p * 6 + _p * _md + _h * 5 + 4 + 2
        assert vec.shape[0] == _attn_row_len, (
            f"attn_comm pack len {vec.shape[0]} != {_attn_row_len} (P={_p},H={_h},msg={_md}); "
            f"robot_obs_dim={self.robot_obs_dim} is full-row / hybrid, not attn-only"
        )
        return vec.copy()

    def _pack_undetermined_obs(self, robot_index, robot, for_feature):
        """All goal positions + local claimed visibility within r; others' target_id; pending flag."""
        px, py = float(robot.px), float(robot.py)
        theta = float(robot.theta)
        K = int(self.num_goal_targets)
        rr = max(float(self.undetermined_obs_goal_radius), 1e-6)
        scale = max(rr, 1.0)
        base = [
            robot.gx - px,
            robot.gy - py,
            robot.v,
            robot.theta,
            for_feature,
            robot.vx_formation,
            robot.vy_formation,
        ]
        tail = []
        cb = self.claimed_by if self.claimed_by is not None else [-1] * K
        for k in range(K):
            tx, ty = self.goal_positions[k]
            dx = float(tx - px)
            dy = float(ty - py)
            dist = float(cal_distance(px, py, tx, ty))
            in_r = 1.0 if dist <= rr + 1e-9 else 0.0
            if in_r > 0.5:
                owner = int(cb[k]) if k < len(cb) else -1
                if owner < 0:
                    cobs = 0.0
                elif owner == robot_index:
                    cobs = 0.5
                else:
                    cobs = 1.0
            else:
                cobs = -1.0
            tail.extend([dx / scale, dy / scale, in_r, cobs])
        norm = float(max(1, K - 1))
        for j in range(self.robot_num):
            if j == robot_index:
                continue
            tail.append(float(self.robots[j].target_id) / norm)
        tail.append(1.0 if bool(getattr(robot, "undetermined_target_pending", False)) else 0.0)
        vec = np.array(base + tail + [px, py], dtype=np.float32)
        assert vec.shape[0] == self.robot_obs_dim + 2
        return vec

    def _pack_undetermined_v2_obs(self, robot_index, robot, for_feature):
        """
        Nearest-M goal slots (body-frame rel + claim visibility) + goal index norm; no (K-1) peer target_id vector.
        Padding uses k_norm=-1 (masked in target head). Pending flag last before px,py.
        """
        px, py = float(robot.px), float(robot.py)
        K = int(self.num_goal_targets)
        M_cfg = int(self.undetermined_v2_goal_slots)
        M = min(M_cfg, K)
        rr = max(float(self.undetermined_obs_goal_radius), 1e-6)
        scale = max(rr, 1.0)
        base = [
            robot.gx - px,
            robot.gy - py,
            robot.v,
            robot.theta,
            for_feature,
            robot.vx_formation,
            robot.vy_formation,
        ]
        dist_idx = []
        for k in range(K):
            tx, ty = self.goal_positions[k]
            dist_idx.append((cal_distance(px, py, tx, ty), k))
        dist_idx.sort(key=lambda t: t[0])
        cb = self.claimed_by if self.claimed_by is not None else [-1] * K
        norm_k = float(max(1, K - 1))
        tail = []
        for slot in range(M_cfg):
            if slot < M:
                _dist, k = dist_idx[slot]
                tx, ty = self.goal_positions[k]
                dx = float(tx - px)
                dy = float(ty - py)
                dist = float(cal_distance(px, py, tx, ty))
                in_r = 1.0 if dist <= rr + 1e-9 else 0.0
                if in_r > 0.5:
                    owner = int(cb[k]) if k < len(cb) else -1
                    if owner < 0:
                        cobs = 0.0
                    elif owner == robot_index:
                        cobs = 0.5
                    else:
                        cobs = 1.0
                else:
                    cobs = -1.0
                k_norm = float(k) / norm_k
                tail.extend([dx / scale, dy / scale, in_r, cobs, k_norm])
            else:
                tail.extend([0.0, 0.0, 0.0, -1.0, -1.0])
        tail.append(1.0 if bool(getattr(robot, "undetermined_target_pending", False)) else 0.0)
        vec = np.array(base + tail + [px, py], dtype=np.float32)
        # v2-only row length; hybrid mode sets robot_obs_dim = v2_core + attn_tail (see compute_undetermined_v2_attn_hybrid_robot_obs_dim).
        _v2_len = 7 + 5 * int(M_cfg) + 1 + 2
        assert vec.shape[0] == _v2_len, (
            f"undetermined v2 pack len {vec.shape[0]} != {_v2_len} "
            f"(M_cfg={M_cfg}); robot_obs_dim={self.robot_obs_dim} (hybrid tail not included in this assert)"
        )
        return vec

    def _pack_undetermined_v2_attn_hybrid_obs(self, robot_index, robot, for_feature):
        """
        Undetermined v2 layout unchanged, then ConsMAC/AttnComm ally+recv+human+obstacle tail (no duplicate self-7).
        Px, py remain the last two scalars for the target head and critic-friendly slicing.
        """
        v2 = self._pack_undetermined_v2_obs(robot_index, robot, for_feature)
        tail = self._attn_comm_tail_vector(robot_index, robot, for_feature, self.attn_comm_P, self.attn_comm_H)
        vec = np.concatenate([v2[:-2], tail, v2[-2:]], dtype=np.float32)
        assert vec.shape[0] == self.robot_obs_dim + 2
        return vec

    def _pack_undetermined_v3_obs(self, robot_index, robot, for_feature):
        """
        Undetermined v3: v2 nearest-M core + one scalar (prev applied target id norm) before px,py,
        then AttnComm tail from --undetermined_v3_comm_* (fixed caps).
        """
        v2 = self._pack_undetermined_v2_obs(robot_index, robot, for_feature)
        prev_scalar = np.array([float(getattr(robot, "undet_prev_tid_norm", -1.0))], dtype=np.float32)
        core_pre_tail = np.concatenate([v2[:-2], prev_scalar], dtype=np.float32)
        tail = self._attn_comm_tail_vector(
            robot_index, robot, for_feature, self.undetermined_v3_comm_P, self.undetermined_v3_comm_H
        )
        vec = np.concatenate([core_pre_tail, tail, v2[-2:]], dtype=np.float32)
        assert vec.shape[0] == self.robot_obs_dim + 2, (
            f"undetermined v3 pack len {vec.shape[0]} != robot_obs_dim+2={self.robot_obs_dim + 2} "
            f"(M={self.undetermined_v2_goal_slots}, P={self.undetermined_v3_comm_P}, H={self.undetermined_v3_comm_H})"
        )
        return vec

    def _pack_robot_obs_row(self, robot_index, robot, for_feature):
        px, py = robot.px, robot.py
        if self.undetermined_goal_assignment:
            if self.undetermined_goal_v3:
                return self._pack_undetermined_v3_obs(robot_index, robot, for_feature)
            if self.undetermined_goal_v2:
                if (
                    str(getattr(self.args, "architecture_mode", "default")) == "attn_undetermined_goal"
                    and self.use_attn_comm_actor
                ):
                    return self._pack_undetermined_v2_attn_hybrid_obs(robot_index, robot, for_feature)
                return self._pack_undetermined_v2_obs(robot_index, robot, for_feature)
            return self._pack_undetermined_obs(robot_index, robot, for_feature)
        if not self.dynamic_goal_assignment:
            if self.use_attn_comm_actor:
                return self._pack_robot_obs_row_attn_comm(robot_index, robot, for_feature)
            return np.array(
                [
                    robot.gx - px,
                    robot.gy - py,
                    robot.v,
                    robot.theta,
                    for_feature,
                    robot.vx_formation,
                    robot.vy_formation,
                    px,
                    py,
                ],
                dtype=np.float32,
            ).copy()
        K = self.num_goal_targets
        N = self.robot_num
        theta = float(robot.theta)
        base = [
            robot.gx - px,
            robot.gy - py,
            robot.v,
            robot.theta,
            for_feature,
            robot.vx_formation,
            robot.vy_formation,
        ]
        tail = []
        if self.dynamic_obs_pack_version == "legacy":
            for k in range(K):
                tx, ty = self.goal_positions[k]
                tail.extend([tx - px, ty - py])
            for k in range(K):
                tail.append(1.0 if self.claimed_by[k] >= 0 else 0.0)
        else:
            M = self.dynamic_slot_m
            Rv = self.dynamic_vis_radius
            dist_idx = []
            for k in range(K):
                tx, ty = self.goal_positions[k]
                dist_idx.append((cal_distance(px, py, tx, ty), k))
            dist_idx.sort(key=lambda t: t[0])
            c = float(np.cos(theta))
            s = float(np.sin(theta))
            for slot in range(M):
                _, k = dist_idx[slot]
                tx, ty = self.goal_positions[k]
                dx = float(tx - px)
                dy = float(ty - py)
                dist = cal_distance(px, py, tx, ty)
                in_r = 1.0 if float(dist) <= Rv else 0.0
                if K > 1:
                    k_norm = float(k) / float(K - 1)
                else:
                    k_norm = 0.0
                bx = c * dx + s * dy
                by = -s * dx + c * dy
                claimed = 1.0 if self.claimed_by[k] >= 0 else 0.0
                tail.extend([in_r, k_norm, bx, by, float(tx), float(ty), claimed])
        norm = float(max(1, K - 1))
        if self.use_neighbor_attn_actor and self.dynamic_obs_pack_version != "legacy":
            others = []
            rvx = float(robot.vx) if robot.vx is not None else 0.0
            rvy = float(robot.vy) if robot.vy is not None else 0.0
            rtid = int(robot.target_id) % K
            for j in range(N):
                if j == robot_index:
                    continue
                oj = self.robots[j]
                dx = float(oj.px - px)
                dy = float(oj.py - py)
                dist = float(cal_distance(px, py, oj.px, oj.py))
                dist_n = dist / 25.0
                ovx = float(oj.vx) if oj.vx is not None else 0.0
                ovy = float(oj.vy) if oj.vy is not None else 0.0
                tid = int(oj.target_id) % K
                if K > 1:
                    tid_n = float(tid) / float(K - 1)
                else:
                    tid_n = 0.0
                same = 1.0 if tid == rtid else 0.0
                others.append((dist, [dx, dy, dist_n, ovx - rvx, ovy - rvy, tid_n, same]))
            others.sort(key=lambda t: t[0])
            P = self.actor_neighbor_p
            for idx in range(P):
                if idx < len(others):
                    tail.extend(others[idx][1])
                else:
                    tail.extend([0.0] * self.neighbor_actor_feat_dim)
        else:
            for j in range(N):
                if j == robot_index:
                    continue
                tail.append(float(self.robots[j].target_id) / norm)
        vec = np.array(base + tail + [px, py], dtype=np.float32)
        assert vec.shape[0] == self.robot_obs_dim + 2
        return vec

    def _undetermined_sync_all_goals(self):
        K = int(self.num_goal_targets)
        for r in self.robots:
            tid = int(r.target_id) % K
            gx, gy = self.goal_positions[tid]
            r.gx, r.gy = gx, gy

    def _undetermined_auction_duplicate_targets(self):
        """Same discrete target: lowest agent index wins; others marked pending for re-selection."""
        K = int(self.num_goal_targets)
        from collections import defaultdict

        groups = defaultdict(list)
        for i in range(self.robot_num):
            ri = self.robots[i]
            if ri.collision or ri.success:
                continue
            tid = int(ri.target_id) % K
            groups[tid].append(i)
        for _tid, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            for lose in sorted(idxs)[1:]:
                self.robots[lose].undetermined_target_pending = True

    def _compute_undetermined_hungarian_shaping(self):
        K = int(self.num_goal_targets)
        pos = np.array([[r.px, r.py] for r in self.robots], dtype=np.float64)
        goals = np.array(self.goal_positions, dtype=np.float64)
        tids = np.array([int(r.target_id) % K for r in self.robots], dtype=np.int64)
        cur = assignment_cost_for_target_ids(pos, goals, tids)
        _opt_ass, opt_total = optimal_assignment_cost(pos, goals)
        gap = float(cur - opt_total)
        scale = float(getattr(self.args, "undetermined_hungarian_reward_scale", 0.15))
        if getattr(self, "undetermined_goal_v2", False) or getattr(self, "undetermined_goal_v3", False):
            div = float(getattr(self.args, "undetermined_v2_hungarian_team_divisor", 8.0))
            div = max(div, 1.0)
            per = -scale * gap / div
        else:
            n = max(1, self.robot_num)
            per = -scale * gap / float(n)
        self.undetermined_hungarian_bonus = np.full(self.robot_num, per, dtype=np.float64)

    def apply_undetermined_targets(self, targets):
        """
        Set each agent's target_id from policy, sync goals, auction duplicates, compute Hungarian shaping bonus.
        targets: length N array-like of ints in [0, K-1].
        """
        K = int(self.num_goal_targets)
        t = np.asarray(targets, dtype=np.int64).reshape(-1)
        assert len(t) == self.robot_num, (len(t), self.robot_num)
        for i, r in enumerate(self.robots):
            r.target_id = int(t[i]) % K
            r.undetermined_target_pending = False
        self._undetermined_sync_all_goals()
        self._undetermined_auction_duplicate_targets()
        self._compute_undetermined_hungarian_shaping()

    def refresh_observations_after_target_change(self):
        """Rebuild per-agent obs like reset() after apply_undetermined_targets (for_feature=0)."""
        obs = []
        for i, robot in enumerate(self.robots):
            temp_obs = np.zeros(((1 + max(self.human_num, self.att_agents)), self.obs_dim + 2))
            temp_obs[0, : self.robot_obs_dim + 2] = self._pack_robot_obs_row(i, robot, 0.0).copy()
            for human in self.humans:
                human.dist2rob = cal_distance(robot.px, robot.py, human.px, human.py)
            self.humans = sorted(self.humans, key=lambda x: x.dist2rob, reverse=True)
            assert self.humans[-1].dist2rob < self.humans[-2].dist2rob, "sort error!"
            for j, human in enumerate(self.humans[: self.human_num]):
                htheta = np.arctan2(human.py, human.px)
                temp_obs[j + 1, : self.human_obs_dim] = np.array(
                    [human.px, human.py, human.vx, human.vy, htheta]
                ).copy()
            obs.append(temp_obs)
        return obs

    def _undetermined_refresh_pending_from_observation(self):
        """Within obs radius r of current target: if claimed by another agent, request re-selection."""
        if self.claimed_by is None:
            return
        K = int(self.num_goal_targets)
        rr = float(self.undetermined_obs_goal_radius)
        for i, r in enumerate(self.robots):
            if r.collision or r.success:
                continue
            tid = int(r.target_id) % K
            gx, gy = self.goal_positions[tid]
            if cal_distance(r.px, r.py, gx, gy) > rr + 1e-9:
                continue
            owner = int(self.claimed_by[tid]) if tid < len(self.claimed_by) else -1
            if owner >= 0 and owner != i:
                r.undetermined_target_pending = True

    def _undetermined_comm_conflict_auction(self):
        """Same target as another agent within comm radius: lowest index keeps target; others pending."""
        K = int(self.num_goal_targets)
        Rc = float(self.undetermined_comm_radius)
        from collections import defaultdict

        groups = defaultdict(list)
        for i in range(self.robot_num):
            r = self.robots[i]
            if r.collision or r.success:
                continue
            tid = int(r.target_id) % K
            groups[tid].append(i)
        for _tid, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            idxs = sorted(idxs)
            conflict = False
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if (
                        cal_distance(self.robots[i].px, self.robots[i].py, self.robots[j].px, self.robots[j].py)
                        <= Rc + 1e-9
                    ):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                continue
            winner = idxs[0]
            for lose in idxs[1:]:
                if lose != winner:
                    self.robots[lose].undetermined_target_pending = True

    def _undetermined_v2_exchange_heuristic(self):
        """
        Optional v2 mode: swap discrete targets for two nearby agents (within exchange radius) when a
        criterion passes (see ``--undetermined_v2_exchange_accept_criterion``):

        - fleet_m: fleet bottleneck M = max_k d(robot_k, goal[target_k]) drops by more than min_gain.
        - pair_max: only the pair (i,j): max(d(i,goal_i),d(j,goal_j)) - max(d(i,goal_j),d(j,goal_i)) > min_gain.
        - cone_mutual_greedy_m: velocity cones + strict mutual improvement; greedy disjoint swaps minimizing M.

        Greedy disjoint pairs ordered by largest gain (fleet or pair, matching the criterion).
        """
        if not getattr(self, "undetermined_v2_exchange", False):
            return
        if self.goal_positions is None:
            return
        self.undetermined_v2_exchange_agent_mask.fill(False)
        self.undetermined_v2_exchange_step_M_gain = 0.0
        self.undetermined_v2_exchange_step_S_gain = 0.0
        K = int(self.num_goal_targets)
        if K < 2:
            return
        crit = str(getattr(self.args, "undetermined_v2_exchange_accept_criterion", "fleet_m"))
        if crit == "cone_mutual_greedy_m":
            self._undetermined_v2_exchange_cone_mutual_greedy_m()
            return
        R_ex = float(self.undetermined_v2_exchange_radius)
        min_gain = float(getattr(self.args, "undetermined_v2_exchange_min_gain", 0.05))
        max_pairs = max(1, int(getattr(self.args, "undetermined_v2_exchange_max_pairs_per_step", 1)))
        ignore_pending = bool(getattr(self.args, "undetermined_v2_exchange_ignore_pending", False))

        def dist_to_assigned_goal(idx: int) -> float:
            r = self.robots[idx]
            if r.collision or r.success:
                return 0.0
            tid = int(r.target_id) % K
            gx, gy = self.goal_positions[tid]
            return cal_distance(r.px, r.py, gx, gy)

        cur_dist = [dist_to_assigned_goal(k) for k in range(self.robot_num)]
        M_fleet = max(cur_dist) if cur_dist else 0.0

        candidates = []
        for i in range(self.robot_num):
            ri = self.robots[i]
            if ri.collision or ri.success:
                continue
            if not ignore_pending and bool(getattr(ri, "undetermined_target_pending", False)):
                continue
            for j in range(i + 1, self.robot_num):
                rj = self.robots[j]
                if rj.collision or rj.success:
                    continue
                if not ignore_pending and bool(getattr(rj, "undetermined_target_pending", False)):
                    continue
                if cal_distance(ri.px, ri.py, rj.px, rj.py) > R_ex + 1e-9:
                    continue
                ti = int(ri.target_id) % K
                tj = int(rj.target_id) % K
                if ti == tj:
                    continue
                gxi, gyi = self.goal_positions[ti]
                gxj, gyj = self.goal_positions[tj]
                d_i_after = cal_distance(ri.px, ri.py, gxj, gyj)
                d_j_after = cal_distance(rj.px, rj.py, gxi, gyi)
                others = 0.0
                for k in range(self.robot_num):
                    if k == i or k == j:
                        continue
                    others = max(others, cur_dist[k])
                M_after = max(others, d_i_after, d_j_after)
                gain_fleet = M_fleet - M_after
                d1 = float(cur_dist[i])
                d2 = float(cur_dist[j])
                gain_pair = max(d1, d2) - max(d_i_after, d_j_after)
                if crit == "pair_max":
                    sort_gain = gain_pair
                    accept = gain_pair > min_gain + 1e-9
                else:
                    sort_gain = gain_fleet
                    accept = gain_fleet > min_gain + 1e-9
                if accept:
                    candidates.append((sort_gain, i, j))

        candidates.sort(key=lambda t: t[0], reverse=True)
        used = set()
        n_swaps = 0
        for _gain, i, j in candidates:
            if i in used or j in used:
                continue
            ri, rj = self.robots[i], self.robots[j]
            ti = int(ri.target_id) % K
            tj = int(rj.target_id) % K
            ri.target_id = tj
            rj.target_id = ti
            ri.undetermined_target_pending = False
            rj.undetermined_target_pending = False
            self.undetermined_v2_exchange_agent_mask[i] = True
            self.undetermined_v2_exchange_agent_mask[j] = True
            used.add(i)
            used.add(j)
            n_swaps += 1
            if n_swaps >= max_pairs:
                break
        if n_swaps > 0:
            S_before = float(sum(cur_dist))
            new_dist = [dist_to_assigned_goal(k) for k in range(self.robot_num)]
            M_after = max(new_dist) if new_dist else 0.0
            S_after = float(sum(new_dist))
            self.undetermined_v2_exchange_step_M_gain = float(M_fleet - M_after)
            self.undetermined_v2_exchange_step_S_gain = float(S_before - S_after)
            self._undetermined_sync_all_goals()
            self._undetermined_auction_duplicate_targets()
            self._compute_undetermined_hungarian_shaping()

    def _dynamic_path_sync_and_conflict_count(self):
        self.dynamic_step_travel_sum = 0.0
        self.dynamic_same_target_conflict_pairs = 0
        K = self.num_goal_targets
        d_thr = float(getattr(self.args, "dynamic_same_target_conflict_dist", 2.0))
        cx, cy = self.goal_centroid_xy
        for i, robot in enumerate(self.robots):
            if robot.prev_px is None:
                robot.prev_px, robot.prev_py = robot.px, robot.py
            if not (robot.success or robot.collision):
                self.dynamic_step_travel_sum += cal_distance(
                    robot.prev_px, robot.prev_py, robot.px, robot.py
                )
            robot.prev_px, robot.prev_py = robot.px, robot.py
        for robot in self.robots:
            if robot.success or robot.collision:
                continue
            tid = int(robot.target_id) % K
            gx, gy = self.goal_positions[tid]
            robot.gx, robot.gy = gx, gy
            robot.for_std = [gx - cx, gy - cy]
        for i in range(self.robot_num):
            for j in range(i + 1, self.robot_num):
                ri, rj = self.robots[i], self.robots[j]
                if ri.collision or rj.collision:
                    continue
                if int(ri.target_id) % K != int(rj.target_id) % K:
                    continue
                if cal_distance(ri.px, ri.py, rj.px, rj.py) < d_thr:
                    self.dynamic_same_target_conflict_pairs += 1

    def _dynamic_process_claims(self):
        K = self.num_goal_targets
        candidates = {k: [] for k in range(K)}
        for i, r in enumerate(self.robots):
            if r.collision or r.success:
                continue
            k = int(r.target_id) % K
            gx, gy = self.goal_positions[k]
            if cal_distance(r.px, r.py, gx, gy) <= r.radius:
                candidates[k].append(i)
        for k, S in candidates.items():
            if not S:
                continue
            gx, gy = self.goal_positions[k]
            if len(S) > 1:
                continue
            i = S[0]
            r = self.robots[i]
            owner = self.claimed_by[k]
            if owner >= 0 and owner != i and self.robots[owner].success:
                r.collision = True
                r.success = False
                self.dynamic_episode_had_collision = True
                continue
            blocked = False
            for j, o in enumerate(self.robots):
                if j == i:
                    continue
                if cal_distance(o.px, o.py, gx, gy) <= o.radius + 1e-5:
                    blocked = True
                    break
            if blocked:
                continue
            if owner >= 0 and owner != i:
                r.collision = True
                r.success = False
                self.dynamic_episode_had_collision = True
                continue
            self.claimed_by[k] = i
            if r.success is not True:
                gate = int(getattr(self.args, "dynamic_arrival_require_outside_entry", 1)) != 0
                if not gate:
                    r.dynamic_just_arrived = True
                elif r.prev_px is None or r.prev_py is None:
                    r.dynamic_just_arrived = True
                else:
                    prev_d = cal_distance(float(r.prev_px), float(r.prev_py), gx, gy)
                    r.dynamic_just_arrived = prev_d > r.radius + 1e-6
            r.success = True
            r.px, r.py = gx, gy
            r.v = 0
            r.vx = 0
            r.vy = 0
        if (self.dynamic_goal_assignment or self.undetermined_goal_assignment) and all(
            self.robots[i].success for i in range(self.robot_num)
        ):
            if not self.dynamic_episode_had_collision and not self.collision_flag:
                self.dynamic_formation_success_once = True

    def set_comm_broadcasts(self, msgs):
        """Last-step intent vectors per robot; shape (robot_num, attn_comm_msg_dim)."""
        arr = np.asarray(msgs, dtype=np.float32).reshape(self.robot_num, self.attn_comm_msg_dim)
        np.copyto(self.agent_broadcast_msg_prev, self.agent_broadcast_msg)
        np.copyto(self.agent_broadcast_msg, arr)

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.robots = [Robot(self.args) for i in range(self.robot_num)]
        self.agent_broadcast_msg.fill(0.0)
        self.agent_broadcast_msg_prev.fill(0.0)
        self._cons_decaf_rf_prev = 0.0
        self._cons_decaf_rv_prev = 0.0
        self.total_obs = []
        # print('env has reset!')
        if self.robots is None:
            raise AttributeError('robots has not set!')
        else:
            self.global_time = 0
                
            #initialize robots
            self.collision_flag = False
            self.laplacian_S_L = float("nan")
            self.laplacian_S_L_prev = float("nan")
            set_bias = np.random.random() * 0.5
            px = -5
            py = -5
            coordinates_list = [(4,6),(3,6),(2,6),(1,6),(0,6),(-1,6),(-2,6),(4,5),(3,5),(1,5),(-1,5),(-5,4),(-5,6)]
            # coordinates_list = [(2,2), (2,1.5), (2,1), (2,0.5)]
            random_index = np.random.randint(len(coordinates_list))
            random_coordinate = coordinates_list[random_index]
            gx = random_coordinate[0] + set_bias
            gy = random_coordinate[1] + set_bias
            # 默认起始位置偏移（保留原有起始点/目标增长逻辑）
            n = 0
            # 如果你想使用自定义的 S 型队形目标点（绝对坐标），在这里列出目标坐标。
            # 下面是用户提供的 S 型目标点列表（10 个点）：
            # candidate_targets = [
            #     [[1, 26] ,[1, 22], [ 1, 18],[ 1, 14],[ 1, 10],[ 1,  6], [ 4, 4], [ 7,  3], [ 10, 4], [ 13, 6], [ 13, 10],[ 13, 14],[ 13, 18], [13,22],[13, 26]], #U
            #     [[1, 26], [3, 26], [5, 26], [7, 26], [9, 26], [11, 26], [13, 26], [7, 15], [7, 12], [7, 9], [7, 6], [7, 3], [7, 17], [7, 20], [7, 23]], #T
            #     [[1,26],[1,22],[1,18],[1,11],[1,7],[1,3],[5,26],[9,26],[13,26],[1,15],[5,15],[9,15],[5,3],[9,3],[13,3]], #E
            #     [[13,25],  [10, 26], [7,26], [4,25],[2,23], [1, 19], [1, 15], [1, 11], [1, 7], [3, 4], [6,3], [8, 3],[10, 3], [12, 4], [13, 6]], #C
            #     [[12, 24], [9, 26], [6, 26], [3, 24], [1, 19], [4, 16], [7, 15], [10, 13], [12, 11], [13, 8], [11, 5], [9, 4], [7, 3], [4,4], [1, 5]], #S
            #     [[1, 26],[1, 22], [1,13], [1,8], [1, 3],[1,18],[4,18],[7,18],[10,18],[13,18], [13, 26],[13, 22], [13,13], [13,8], [13,3]], #H
            # ]
            candidate_targets = [
                [[1, 26] , [ 1, 18],[ 1, 12],[ 1,  6], [ 5, 3], [ 9, 3], [ 13, 6], [ 13, 12],[ 13, 18], [13, 26]]  ,      #U
 [[1, 26], [4, 26], [7, 26], [10, 26], [13, 26], [7, 23], [7, 18], [7, 13], [7, 8], [7, 3]]                        ,      #T                                       
 [[1,26],[1,18],[1,11],[1,3],[7,26],[13,26],[5,15],[9,15],[7,3],[13,3]]                                            ,      #E           
 [[12, 25], [7,26], [3,24], [1, 19], [1, 15],  [1, 10], [3, 4], [6,3], [10, 3],  [13, 6]]                          ,      #C               
[[12, 24], [7, 26], [3, 24], [1, 19], [5, 15],  [11, 12],  [13, 8], [10, 4],  [6,4], [1, 5]]                       ,      #S                               
 [[1, 26],[1, 11], [1, 3],[1,19],[5,18],[9,18],[13,19], [13, 26],[13, 11], [13,3]]                                 ,      #H                           
 [[7,26],[7,16],[2,16],[12,16],[4,20], [10, 20],[1,11],[1,3],[13,11],[13,3]]                                       ,      #A                                                          
 [[3, 26], [1, 21], [1, 15], [1,9], [1,3], [8, 26], [7,15],[7,3], [12, 21],[13,9]]                                 ,      #B           
[[3, 26], [1, 21], [1, 15], [1,9], [1,3], [8, 26], [13,15],[7,3], [12, 21],[11,7]]                                 ,      #D                                                                                                         
[[1,26],[1,21],[1,15],[1,9], [1,3],[5,26],[9,26],[13,26],[5,15],[9,15]]                                            ,      #F               
[ [8,26], [3,24], [1, 17],   [1, 10], [5, 3], [9, 3],  [13, 7],[13, 12],[9, 12],[13,3], ]                          ,      #G                               
 [[7,26],[7,21], [7,16],[7,11],[7,7],[7,3],[6, 26],[8,26],[5,3],[9,3]]                                             ,      #I                       
 [ [10,26],[13, 26], [13,22],  [13, 18], [13, 14],  [13, 10], [3, 4], [6,3], [10, 3],  [13, 6],]                   ,      #J                                                   
 [[1,26],[1,20], [1,15],[1,10],[1,3],[10,26], [6, 19],[5, 12],[9, 8],[13,3]]                                       ,      #K                               
 [[1,26],[1,21], [1,16],[1,11],[1,7],[1,3], [4,3],[7,3],[10,3],[13,3]]                                             ,      #L               
 [[1,3],[1,17],[3, 26],[5,26], [7,15],[7,5],[9,26],[11,26],[13,17],[13,3]]                                         ,      #M                                   
[[13,26],[12, 19], [11, 11],[10, 3], [8,11],[6,19],[4,26],[3, 19],[2,11],[1,3]]                                    ,      #N                                                                             
[[5, 26], [2, 21], [1, 15], [2,9], [9,3], [9, 26], [13,15],[5,3], [12, 21],[12,9]]                                 ,      #O               
[[1, 26], [1, 21], [1, 15], [1,9], [1,3],[6,26], [10, 26], [13,22],[10,17],[6,17]]                                 ,      #P                           
[ [3, 20],  [2,11], [9,3], [7, 26], [5,3], [11, 20],[12, 11],[8,8], [11,6] ,[13,3]]                                ,      #Q               
[[1, 26], [1, 18],  [1,10], [1,3],[6,26], [10, 26], [12,21],[6,17],[8,11],[13,3]]                                  ,      #R                                       
[[1, 26] , [ 1, 21],[ 1, 16],[ 4,  11], [ 7, 3], [ 7, 6], [ 10, 11], [ 13, 16],[ 13, 21], [13, 26]]                ,      #V                                                           
[[1,26],[1,12],[3, 3],[5,3], [7,14],[7,24],[9,3],[11,3],[13,12],[13,26]]                                           ,      #W                       
[[1,26],[4, 21],[7,13],[10,9],[13,3],[1,3],[4,9],[7,17],[10,21],[13,26]]                                           ,      #X                   
[[1, 26] , [ 3, 21],[ 5, 16], [ 7, 3], [ 7, 6], [7,9],[7,12], [ 9, 16],[ 11, 21], [13, 26]]                        ,      #Y               
[[1,26], [5,26],[9,26],[13,26],[9,19],[5,11],[1,3],[5,3],[9,3],[13,3]]                                             ,      #Z           
            ]                                                                                                                                               
            # s_shape_targets = candidate_targets[np.random.randint(0, len(candidate_targets))]
            s_shape_targets = self.s_shape_targets
            # s_shape_targets = candidate_targets[ [ 1,4][np.random.randint(0, 2)]]
            # s_shape_targets = candidate_targets[ [0, 1, 2, 4][np.random.randint(0, 2)]]
            # s_shape_targets = candidate_targets[ [0, 1,2,4 ,3,5,6,7][np.random.randint(0, 8)]]
            # s_shape_targets = candidate_targets[list(range(0,16))[np.random.randint(0, 16)]]
            # 选择用于赋值的目标点数目（如果机器人数量少于点，截断；多于点，重复最后一个点）
            used_targets = []
            if self.robot_num <= len(s_shape_targets):
                used_targets = s_shape_targets[: self.robot_num]
            else:
                used_targets = list(s_shape_targets)
                # 若机器人数量超过目标点数，则重复最后一个点作为占位
                while len(used_targets) < self.robot_num:
                    used_targets.append(s_shape_targets[-1])

            # 将目标点转换为相对于质心的相对模板（for_std），以便队形对平移不敏感
            centroid = np.mean(np.array(used_targets), axis=0)
            # used_targets = sorted(used_targets, key= lambda target: ((np.arctan(1/np.divide(*(target-centroid))) + (np.pi if (target-centroid)[0]<0 else 0))*2/np.pi+4)%4)
            rel_targets = [(tx - centroid[0], ty - centroid[1]) for (tx, ty) in used_targets]

            rand_pos = None
            if getattr(self.args, "randomize_robot_initial_positions", False):
                _spawn_mode = str(getattr(self.args, "robot_initial_spawn_mode", "random_box"))
                if _spawn_mode == "cluster_comm":
                    rand_pos = self._sample_clustered_robot_starts_in_comm_range()
                elif _spawn_mode == "cluster_disk":
                    rand_pos = self._sample_clustered_robot_starts_in_disk()
                else:
                    rand_pos = self._sample_collision_free_robot_starts()

            if self.dynamic_goal_assignment or self.undetermined_goal_assignment:
                self.goal_positions = [
                    (gx + rel_targets[j][0], gy + rel_targets[j][1])
                    for j in range(len(rel_targets))
                ]
                gcx = float(np.mean([p[0] for p in self.goal_positions]))
                gcy = float(np.mean([p[1] for p in self.goal_positions]))
                self.goal_centroid_xy = (gcx, gcy)
                self.num_goal_targets = len(self.goal_positions)
                K = self.num_goal_targets
                self.claimed_by = [-1] * K
                self.dynamic_formation_success_once = False
                self.dynamic_episode_had_collision = False
                self.undetermined_hungarian_bonus = np.zeros(self.robot_num, dtype=np.float64)
                self.undetermined_v2_fleet_M_prev = float("nan")
                self.undetermined_v2_fleet_S_prev = float("nan")
                self.undetermined_v2_fleet_M_drop = 0.0
                self.undetermined_v2_fleet_S_drop = 0.0
                self.undetermined_v2_exchange_step_M_gain = 0.0
                self.undetermined_v2_exchange_step_S_gain = 0.0
                self.undetermined_v2_exchange_agent_mask = np.zeros(self.robot_num, dtype=np.bool_)
            else:
                K = len(rel_targets)

            px_cursor, py_cursor = px, py
            for i, robot in enumerate(self.robots):
                robot.vx_formation = 0
                robot.vy_formation = 0
                if rand_pos is not None:
                    px_i, py_i = rand_pos[i]
                else:
                    px_i, py_i = px_cursor, py_cursor
                    px_cursor += robot.edge * np.cos(2 * np.pi * i / 10)
                    py_cursor += robot.edge * np.sin(2 * np.pi * i / 10)
                if self.dynamic_goal_assignment:
                    tid = i % K
                    gxi, gyi = self.goal_positions[tid]
                    robot.set(px_i, py_i, gxi, gyi, 0, 0, np.pi / 2)
                    robot.target_id = int(tid)
                    robot.prev_target_id = int(tid)
                    robot.target_switched_this_step = False
                    robot.prev_px = float(px_i)
                    robot.prev_py = float(py_i)
                    robot.for_std = [gxi - self.goal_centroid_xy[0], gyi - self.goal_centroid_xy[1]]
                elif self.undetermined_goal_assignment:
                    tid = i % K
                    gxi, gyi = self.goal_positions[tid]
                    robot.set(px_i, py_i, gxi, gyi, 0, 0, np.pi / 2)
                    robot.target_id = int(tid)
                    robot.prev_target_id = int(tid)
                    robot.target_switched_this_step = False
                    robot.prev_px = float(px_i)
                    robot.prev_py = float(py_i)
                    robot.for_std = [gxi - self.goal_centroid_xy[0], gyi - self.goal_centroid_xy[1]]
                    robot.undetermined_target_pending = True
                else:
                    robot.set(px_i, py_i, gx, gy, 0, 0, np.pi / 2)
                    robot.gx = gx + rel_targets[i][0]
                    robot.gy = gy + rel_targets[i][1]
                    robot.for_std = [rel_targets[i][0], rel_targets[i][1]]
                n += 1
                robot.v = 0
                robot.theta = 0
                robot.dmin = float('inf')
                robot.id = 0
                robot.collision = None
                robot.success = None
                robot.dynamic_just_arrived = False
                robot.dynamic_hold_target_steps = 0
                robot.dynamic_hold_at_switch = 0
                robot.dynamic_episode_target_switch_count = 0
                robot.dynamic_episode_target_switch_prior = 0
                robot.dist_nearest_unclaimed = None
                robot.pre_dist_nearest_unclaimed = None
                robot.dist_unc_near_cluster = None
                robot.pre_dist_unc_near_cluster = None
                robot.dist_unclaimed_outside_near = None
                robot.pre_dist_unclaimed_outside_near = None
                robot.dynamic_loiter_near_accum = 0
                robot.dynamic_loiter_at_switch = 0
            
            for robot in self.robots:
                for r in self.robots:
                    if robot != r:
                        robot.vx_formation -= (robot.px - r.px - (robot.for_std[0] - r.for_std[0]))
                        robot.vy_formation -= (robot.py - r.py - (robot.for_std[1] - r.for_std[1]))

            if self.dynamic_goal_assignment or self.undetermined_goal_assignment:
                _td0 = 0.0
                for _r in self.robots:
                    if _r.collision is True:
                        continue
                    _td0 += float(cal_distance(_r.px, _r.py, _r.gx, _r.gy))
                self.dynamic_team_dist_sum_prev = _td0
                self.dynamic_team_dist_sum_this_step = _td0
            
            # W = (np.ones((self.robot_num, self.robot_num)) - np.eye(self.robot_num)) * self.for_edge
            
            # Build adjacency/weight matrix using the actual targets assigned to robots
            # (used_targets length equals number of robots considered). Using s_shape_targets
            # here caused L_des to have a size mismatch when robot_num != len(s_shape_targets).
            W = np.array([[get_weight(*c1, *c2) for c1 in used_targets] for c2 in used_targets])

            # Degree matrix: diagonal of row-sums of W
            row_sums = np.sum(W, axis=1)
            D = np.diag(row_sums)
            L = D - W
            D_sys = np.diag(np.power(row_sums, -1.0 / 2.0))  # symmetric normalize
            self.L_des = D_sys @ L @ D_sys
            assert np.round(self.L_des[0][0]) == 1,'L compute error!'

            #原本是根据不同的case初始化障碍物
            self.generate_random_human_position(human_num=self.human_num, rule=self.train_val_sim)
            self.generate_remote_human_position()

            #设置时间步长
            for agent in self.robots + self.humans :
                agent.time_step = self.time_step
            
            obs = []
            temp_robot_obs = []
            temp_human_obs = []
            for i,robot in enumerate(self.robots):
                temp_obs = np.zeros(((1 + max(self.human_num, self.att_agents)), self.obs_dim+2))
                #TODO formation compute
                temp_obs[0,:self.robot_obs_dim+2] = self._pack_robot_obs_row(i, robot, 0.0).copy()

                temp_robot_obs.append(
                np.array([robot.px, robot.py, robot.gx, robot.gy, robot.v, robot.theta, 0, robot.vx_formation, robot.vy_formation]).copy())
                
                for human in self.humans:
                    human.dist2rob = cal_distance(robot.px, robot.py, human.px, human.py)
                self.humans = sorted(self.humans, key=lambda x:x.dist2rob, reverse=True)   #行人距离机器人的距离由远到近进行排序
                assert self.humans[-1].dist2rob < self.humans[-2].dist2rob, 'sort error!'

                for j, human in enumerate(self.humans[:self.human_num]):
                    htheta = np.arctan2(human.py, human.px)
                    temp_obs[j + 1,:self.human_obs_dim] = np.array([human.px, human.py, human.vx, human.vy, htheta]).copy()
                obs.append(temp_obs)
                

                for human in self.humans:
                    temp_human_obs.append(np.array([human.px, human.py, human.vx, human.vy, htheta]).copy())

            self.total_obs.append([temp_robot_obs, temp_human_obs])
            
            obs_orca = []
            for agent in self.robots + self.humans:
                obs_orca.append(np.array([agent.px, agent.py, agent.vx, agent.vy, agent.gx, agent.gy]))

        if self.method == 'ppo':
            return obs
        elif self.method == 'orca' or self.method == 'apf':
            return obs_orca


    def step(self):
        sub_agent_obs = []
        sub_agent_obs_render = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        human_obs = []
        human_actions = []
        id_index = 0   #for formation encode
        reachgoal_num = 0   #for calculate the success rate

        # v3: prev channel = normalized target_id carried into this step (end of last step / post-apply).
        if getattr(self, "undetermined_goal_v3", False):
            K = max(1, int(self.num_goal_targets))
            den = float(max(1, K - 1))
            for r in self.robots:
                r.undet_prev_tid_norm = float(int(r.target_id) % K) / den

        #human action
        for human in self.humans:
             # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            #这里得到的ob是JointState中的human_states，即其它human的状态
            human_action = human.act(ob)
            human_actions.append(human_action)    #human到终点后位置不再更新
            human.theta = np.arctan2(human.vy,human.vx)
            human_obs.append(np.array([human.px, human.py, human.vx, human.vy, human.theta]))
        # for i, human_action in enumerate(human_actions):
        #     self.humans[i].step(human_action)

        if self.dynamic_goal_assignment or self.undetermined_goal_assignment:
            self._dynamic_path_sync_and_conflict_count()

        #for reward calculate
        W = np.zeros((self.robot_num, self.robot_num))
        for i, robot in enumerate(self.robots):
            #initialize some flags of robots
            robot.dmin = float('inf')
            robot.collision = False
            robot.vx_formation = 0
            robot.vy_formation = 0

            for j, car in enumerate(self.robots):
                W[i][j] = get_weight(robot.px, robot.py, car.px, car.py)

            # collision  detect
            for agent in self.robots + self.humans:
                d = cal_distance(robot.px, robot.py, agent.px, agent.py)
                if agent != robot and d < robot.dmin:
                    robot.dmin = cal_distance(robot.px, robot.py, agent.px, agent.py)
                    
            # if robot.dmin - robot.radius - agent.radius < 0 or robot.px < 0 or robot.py < 0: # with boundary
            if robot.dmin - robot.radius - agent.radius < 0:   # without boundary
                robot.collision = True
                # print('collision')
                robot.success = False
                self.collision_flag = True
                if self.dynamic_goal_assignment or self.undetermined_goal_assignment:
                    self.dynamic_episode_had_collision = True
            else:
                robot.dmin = robot.dmin - robot.radius - agent.radius

            #formation detect
            id_index += 1
            robot.id = id_index

            for robot in self.robots:
                for r in self.robots:
                    if robot != r:
                        robot.vx_formation -= (robot.px - r.px - (robot.for_std[0] - r.for_std[0]))
                        robot.vy_formation -= (robot.py - r.py - (robot.for_std[1] - r.for_std[1]))
            # print('robot.px,robot.py',robot.px,robot.py)

        if self.dynamic_goal_assignment or self.undetermined_goal_assignment:
            self._dynamic_process_claims()
        if self.undetermined_goal_assignment:
            self._undetermined_refresh_pending_from_observation()
            self._undetermined_comm_conflict_auction()
            self._undetermined_v2_exchange_heuristic()

        assert W[-1][-2] != 0,'W compute error!'

        # print(W)
        D = np.diag(np.sum(W, axis=1))  # degree matrix via row sums
        L = D - W
        # symmetric normalization: D_sys = diag(row_sums^{-1/2}); guard against zeros
        row_sums = np.sum(W, axis=1)
        eps = 1e-8
        inv_sqrt = np.power(np.where(row_sums > 0, row_sums, eps), -0.5)
        D_sys = np.diag(inv_sqrt)
        L_hat = D_sys @ L @ D_sys  # scaled Laplacian

        # Ensure self.L_des has the same shape as L_hat. If not, rebuild L_des from current robot goals.
        if (self.L_des is None) or (self.L_des.shape != L_hat.shape):
            # Build W_des from current robots' goal positions (gx, gy)
            try:
                if self.dynamic_goal_assignment or self.undetermined_goal_assignment:
                    targets = list(self.goal_positions)
                else:
                    targets = [(r.gx, r.gy) for r in self.robots]
                W_des = np.array([[get_weight(tx1, ty1, tx2, ty2) for (tx1, ty1) in targets] for (tx2, ty2) in targets])
                row_sums_des = np.sum(W_des, axis=1)
                inv_sqrt_des = np.power(np.where(row_sums_des > 0, row_sums_des, eps), -0.5)
                D_des = np.diag(row_sums_des)
                L_des = D_des - W_des
                D_sys_des = np.diag(inv_sqrt_des)
                self.L_des = D_sys_des @ L_des @ D_sys_des
            except Exception:
                # If rebuilding fails for any reason, raise a clear error with shapes for debugging
                raise RuntimeError(f"L_des shape mismatch and rebuild failed: L_hat.shape={L_hat.shape}, L_des.shape={(None if self.L_des is None else self.L_des.shape)}")

        for_feature = np.trace(np.transpose((L_hat - self.L_des)) @ (L_hat - self.L_des))
        _nf = float(np.sqrt(max(float(for_feature), 0.0)))
        _den = float(np.linalg.norm(self.L_des, ord="fro"))
        try:
            _cur_sl = float(self.laplacian_S_L)
            self.laplacian_S_L_prev = _cur_sl if math.isfinite(_cur_sl) else float("nan")
        except (TypeError, ValueError, AttributeError):
            self.laplacian_S_L_prev = float("nan")
        self.laplacian_S_L = float(1.0 - _nf / _den) if _den > 1e-12 else float("nan")

        #navigation
        for robot in self.robots:
            if cal_distance(robot.px,robot.py,robot.gx,robot.gy) <= robot.radius:
                reachgoal_num += 1
            
            robot.pre_dist2goal = robot.dist2goal
            robot.dist2goal = cal_distance(robot.px,robot.py,robot.gx,robot.gy)

        if getattr(self, "undetermined_v2_exchange", False):
            cur_dists = []
            s_sum = 0.0
            for robot in self.robots:
                if robot.collision or robot.success:
                    d = 0.0
                else:
                    d = float(robot.dist2goal) if robot.dist2goal is not None else 0.0
                cur_dists.append(d)
                s_sum += d
            cur_M = max(cur_dists) if cur_dists else 0.0
            prev_M = float(getattr(self, "undetermined_v2_fleet_M_prev", float("nan")))
            prev_S = float(getattr(self, "undetermined_v2_fleet_S_prev", float("nan")))
            if math.isfinite(prev_M):
                self.undetermined_v2_fleet_M_drop = max(0.0, prev_M - cur_M)
            else:
                self.undetermined_v2_fleet_M_drop = 0.0
            if math.isfinite(prev_S):
                self.undetermined_v2_fleet_S_drop = max(0.0, prev_S - s_sum)
            else:
                self.undetermined_v2_fleet_S_drop = 0.0
            self.undetermined_v2_fleet_M_prev = cur_M
            self.undetermined_v2_fleet_S_prev = s_sum
        else:
            self.undetermined_v2_fleet_M_drop = 0.0
            self.undetermined_v2_fleet_S_drop = 0.0

        if (self.dynamic_goal_assignment or self.undetermined_goal_assignment) and self.claimed_by is not None:
            cb = self.claimed_by
            K = self.num_goal_targets
            unclaimed = [k for k in range(K) if int(cb[k]) < 0]
            for robot in self.robots:
                robot.pre_dist_nearest_unclaimed = getattr(
                    robot, "dist_nearest_unclaimed", None
                )
                if not unclaimed:
                    robot.dist_nearest_unclaimed = None
                else:
                    bd = float("inf")
                    for kk in unclaimed:
                        gx, gy = self.goal_positions[kk]
                        d = cal_distance(robot.px, robot.py, gx, gy)
                        if d < bd:
                            bd = d
                    robot.dist_nearest_unclaimed = float(bd)

        if self.dynamic_goal_assignment:
            thr_l = float(getattr(self.args, "dynamic_loiter_goal_dist_thresh", 2.5))
            for robot in self.robots:
                if robot.collision or robot.success:
                    robot.dynamic_loiter_near_accum = 0
                    robot.dynamic_loiter_at_switch = 0
                    continue
                sw = bool(getattr(robot, "target_switched_this_step", False))
                if sw:
                    robot.dynamic_loiter_at_switch = int(getattr(robot, "dynamic_loiter_near_accum", 0))
                    robot.dynamic_loiter_near_accum = 0
                else:
                    robot.dynamic_loiter_at_switch = 0
                    if robot.dist2goal is not None and float(robot.dist2goal) < thr_l:
                        robot.dynamic_loiter_near_accum = int(getattr(robot, "dynamic_loiter_near_accum", 0)) + 1
                    else:
                        robot.dynamic_loiter_near_accum = 0

            self.reward_calculator.begin_dynamic_reward_step(self, for_feature)

        if self.dynamic_goal_assignment:
            _td = 0.0
            for _r in self.robots:
                if _r.collision is True:
                    continue
                _td += float(cal_distance(_r.px, _r.py, _r.gx, _r.gy))
            self.dynamic_team_dist_sum_this_step = _td

        self.global_time += self.time_step
        reward = 0
        for i, robot in enumerate(self.robots):
            sub_agent_obs.append(self.get_obs(robot, self.humans, for_feature)[0])
            sub_agent_obs_render.append(self.get_obs(robot, self.humans, for_feature)[1])
            sub_agent_reward.append(self.get_reward(robot,for_feature))
            # reward += self.get_reward(robot,for_feature)

            sub_agent_done.append(self.get_done(robot))
            sub_agent_info.append(self.get_info(robot))

        if self.dynamic_goal_assignment:
            self.dynamic_team_dist_sum_prev = self.dynamic_team_dist_sum_this_step
            self.dynamic_formation_success_once = False
            for _r in self.robots:
                _r.target_switched_this_step = False
                _r.dynamic_just_arrived = False

        # sub_agent_reward = [reward] * len(self.robots)
        sub_agent_obs = np.array(sub_agent_obs)
        #for visualize(all step obs)
        self.total_obs.append([sub_agent_obs_render, human_obs])  #dim(step, 2, agent_num, obs_dim)

        if self.collision_flag == True:
            sub_agent_done = [True] * self.robot_num

        obs_orca = []
        for agent in self.robots + self.humans:
            obs_orca.append(np.array([agent.px, agent.py, agent.vx, agent.vy, agent.gx, agent.gy]))

        if self.method == 'ppo':
            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
        if self.method == 'orca' or self.method == 'apf':
            return  [obs_orca]
    
    def get_obs(self, robot, humans, for_feature):
        obs = np.zeros(((1 + max(self.human_num, self.att_agents)), self.obs_dim+2))

        # px = robot.px + self.time_step * robot.v * np.cos(robot.theta)
        # py = robot.py + self.time_step * robot.v * np.sin(robot.theta)
        px = robot.px
        py = robot.py
        gx = robot.gx - px
        gy = robot.gy - py

        v = robot.v
        theta = robot.theta
        vx_for = robot.vx_formation
        vy_for = robot.vy_formation

        ri = self.robots.index(robot)
        obs[0,:self.robot_obs_dim+2] = self._pack_robot_obs_row(ri, robot, for_feature)
        obs_render = np.array([px, py, gx, gy, v, theta, for_feature, vx_for, vy_for])

        for human in humans:
            human.dist2rob = cal_distance(robot.px, robot.py, human.px, human.py)
        humans = sorted(humans, key=lambda x:x.dist2rob, reverse=True)   #行人距离机器人的距离由远到近进行排序
        assert humans[-1].dist2rob <= humans[-2].dist2rob, 'sort error!'

        for j,human in enumerate(humans[:self.human_num]):
            obs[j + 1,:self.human_obs_dim] = \
            np.array([human.px, human.py, human.vx, human.vy, np.arctan2(human.py,human.px)]).copy()
        return [obs, obs_render]
    
    def get_reward(self,robot,for_feature):
        # Keep EnvCore.get_reward() signature, but delegate computation.
        return self.reward_calculator.compute_default_reward(robot, for_feature)
    
    def get_done(self,agent):
        if self.undetermined_goal_assignment:
            if agent.collision == True:
                return True
            if agent.success == True:
                return True
            if self.global_time >= self.time_limit:
                return True
            return False
        if self.dynamic_goal_assignment:
            if agent.collision == True:
                return True
            if agent.success == True:
                return True
            if self.global_time >= self.time_limit:
                return True
            return False
        done = False
        if agent.collision == True:
            done = True

        if reach_goal(agent):
            done = True

        if self.global_time >= self.time_limit:
            done = True

        return done
    
    def _attach_reward_terms(self, agent, info):
        t = getattr(agent, "_reward_terms", None)
        if t is not None:
            info.reward_terms = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in t.items()}
        else:
            info.reward_terms = {}
        if self.undetermined_goal_assignment:
            setattr(info, "undetermined_need_target", bool(getattr(agent, "undetermined_target_pending", False)))
        return info

    def get_info(self, agent):
        if self.undetermined_goal_assignment:
            if agent.collision == True:
                return self._attach_reward_terms(agent, Collision())
            if self.global_time >= self.time_limit:
                return self._attach_reward_terms(agent, Timeout())
            if agent.dmin < agent.discomfort_dist:
                return self._attach_reward_terms(agent, Danger())
            if agent.success == True:
                return self._attach_reward_terms(agent, ReachGoal())
            return self._attach_reward_terms(agent, Nothing())
        if self.dynamic_goal_assignment:
            if agent.collision == True:
                return self._attach_reward_terms(agent, Collision())
            if self.global_time >= self.time_limit:
                return self._attach_reward_terms(agent, Timeout())
            if agent.dmin < agent.discomfort_dist:
                return self._attach_reward_terms(agent, Danger())
            if agent.success == True:
                return self._attach_reward_terms(agent, ReachGoal())
            return self._attach_reward_terms(agent, Nothing())
        info = Nothing()
        if agent.collision == True:
            info = Collision()

        if agent.dmin < agent.discomfort_dist:
            info = Danger()

        if self.global_time >= self.time_limit:
            info = Timeout()

        if reach_goal(agent):
            info = ReachGoal()

        return self._attach_reward_terms(agent, info)
    
    
