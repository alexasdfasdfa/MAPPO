"""
BACKUP ONLY — do not import in training code.

Snapshot of RewardCalculator undetermined v2 methods before literal-relax / efficiency edits (2026-04-22).
Copy the two method bodies below into class RewardCalculator in reward_calculator.py (replace existing methods).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class _BackupRewardCalculatorV2Legacy_20260422_DO_NOT_IMPORT:
    """Placeholder class so this file is valid Python; methods are copies for manual restore."""

    env: Any

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
        r = 0.0
        if scale_dense > 1e-12:
            r += scale_dense * max(0.0, min(1.0, sl))
        sl_prev = getattr(env, "laplacian_S_L_prev", float("nan"))
        d_sl = 0.0
        try:
            sp = float(sl_prev)
            if math.isfinite(sp):
                d_sl = max(0.0, sl - sp)
        except (TypeError, ValueError):
            d_sl = 0.0
        r_delta = 0.0
        if scale_delta > 1e-12 and d_sl > 1e-12:
            r_delta = scale_delta * d_sl
            r += r_delta
        succ = 1.0 if sl >= thr else 0.0
        if scale_succ > 1e-12 and sl >= thr:
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
        r_nav += prog_delta * pc

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
            r_nav += approach_term

        r_sl, sl_v, sl_succ, sl_delta = self._undetermined_v2_sl_reward_raw(env, a)
        r_nav += r_sl

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
