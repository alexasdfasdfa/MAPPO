"""
Minimum-cost bipartite matching (Hungarian). Cost matrix shape (n, m), n<=m or pad square.
Uses scipy (CPU); cost can be built from torch then .cpu().numpy().
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def hungarian_min_sum(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    :param cost: 2D array C[i,j] = cost to assign row i to column j.
    :return: (row_ind, col_ind, total_cost) for optimal assignment.
    """
    from scipy.optimize import linear_sum_assignment

    c = np.asarray(cost, dtype=np.float64)
    r, j = linear_sum_assignment(c)
    return r, j, float(c[r, j].sum())


def assignment_cost_for_target_ids(
    positions: np.ndarray,
    goal_xy: np.ndarray,
    target_ids: np.ndarray,
) -> float:
    """Sum of distances robot_i -> goal[target_ids[i]]."""
    n = positions.shape[0]
    s = 0.0
    for i in range(n):
        k = int(target_ids[i])
        dx = float(positions[i, 0]) - float(goal_xy[k, 0])
        dy = float(positions[i, 1]) - float(goal_xy[k, 1])
        s += float(np.hypot(dx, dy))
    return s


def optimal_assignment_cost(positions: np.ndarray, goal_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Square cost matrix dist(robot_i, goal_j); requires n<=K; pad if n>K not supported.
    Returns (col_indices_for_rows, optimal_total_cost).
    """
    n = int(positions.shape[0])
    k = int(goal_xy.shape[0])
    assert n <= k, "need at least as many goals as robots for one-to-one matching"
    c = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            c[i, j] = float(
                np.hypot(
                    positions[i, 0] - goal_xy[j, 0],
                    positions[i, 1] - goal_xy[j, 1],
                )
            )
    r, col, total = hungarian_min_sum(c)
    # col[j] for each row r - scipy returns pairs
    order = np.argsort(r)
    col_sorted = col[order]
    return col_sorted, total
