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

    - `compute_default_reward` preserves current EnvCore logic exactly.
    - `compute_pattern_speed_direction_reward` is a placeholder for future reward shaping.
    """

    def __init__(self, env_core: Any):
        self.env = env_core

    def compute_default_reward(self, robot: Any, for_feature: float) -> np.ndarray:
        """
        Preserve EnvCore.get_reward() current logic.
        """
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

        # navigation
        r_nav += (robot.pre_dist2goal - robot.dist2goal) * 5

        if r_nav > 0 and robot.collision:
            r_nav = 0

        if reach_goal(robot):
            robot.goal_flag = True
            if robot.v != 0:
                r_goal += 5

        if self.env.collision_flag:
            r_goal = 0

        discount_formation = 0
        discount_avoid = 50
        discount_nav = 20
        discount_goal = 200
        discount_bonus = 2

        reward = (
            discount_formation * r_formation
            + discount_avoid * r_avoid
            + discount_nav * r_nav
            + discount_goal * r_goal  # + discount_bonus * r_bonus
        )
        if robot.goal_flag:
            reward = 0

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

