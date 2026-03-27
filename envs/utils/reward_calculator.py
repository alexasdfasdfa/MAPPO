import numpy as np

from dataclasses import dataclass
from typing import Optional, Any

from envs.utils.utils import reach_goal


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

    - `compute_default_reward` uses non-dynamic shaping from config (`nd_*` args); dynamic mode unchanged.
    - `compute_pattern_speed_direction_reward` is a placeholder for future reward shaping.
    """

    def __init__(self, env_core: Any):
        self.env = env_core

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

        discount_formation = 0
        discount_avoid = 50
        discount_nav = 20
        discount_goal = 200
        discount_bonus = 2

        c_formation = discount_formation * r_formation
        c_avoid = discount_avoid * r_avoid
        c_nav = discount_nav * r_nav
        c_goal = discount_goal * r_goal

        reward_shaped = c_formation + c_avoid + c_nav + c_goal
        reward = reward_shaped
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
                    reward = reward + tr
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
            "c_avoid": float(c_avoid),
            "c_nav": float(c_nav),
            "c_goal": float(c_goal),
            "reward_shaped": float(reward_shaped),
            "nd_terminal_tr": float(tr),
            "nd_tr_applied": float(nd_tr_applied),
            "goal_entered_from_outside": float(1.0 if entered else 0.0),
            "goal_flag": float(1.0 if robot.goal_flag else 0.0),
            "reward_final": float(reward),
        }

        return np.array([reward])



    def _compute_dynamic_goal_reward(self, robot: Any, for_feature: float) -> np.ndarray:
        """Team path-length term, same-target proximity penalty, optional success bonus."""
        a = self.env.args
        n = max(1, self.env.robot_num)
        shared = 0.0
        shared -= float(getattr(a, "dynamic_path_reward_scale", 1.0)) * self.env.dynamic_step_travel_sum / n
        npairs = self.env.dynamic_same_target_conflict_pairs
        if npairs > 0:
            shared -= float(getattr(a, "dynamic_same_target_penalty_scale", 2.0)) * npairs / n
        if getattr(robot, "target_switched_this_step", False):
            shared -= float(getattr(a, "dynamic_target_switch_penalty", 0.05))
        if self.env.dynamic_formation_success_once:
            shared += float(getattr(a, "dynamic_formation_success_bonus", 50.0)) / n

        r_avoid = 0.0
        if robot.collision == True:
            r_avoid = -60.0
        else:
            if robot.dmin < robot.discomfort_dist * 2:
                r_avoid = -np.exp(-robot.dmin / 3)

        r_nav = 0.0
        if robot.pre_dist2goal is not None:
            r_nav = (robot.pre_dist2goal - robot.dist2goal) * 5
        if r_nav > 0 and robot.collision:
            r_nav = 0.0

        discount_avoid = 50
        discount_nav = 20
        c_avoid = discount_avoid * r_avoid
        c_nav = discount_nav * r_nav
        reward = shared + c_avoid + c_nav
        robot._reward_terms = {
            "reward_mode": "dynamic",
            "shared_team": float(shared),
            "r_avoid_raw": float(r_avoid),
            "r_nav_raw": float(r_nav),
            "c_avoid": float(c_avoid),
            "c_nav": float(c_nav),
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
