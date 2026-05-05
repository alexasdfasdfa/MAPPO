"""
Spatial clustering + per-agent roles for CTDE dynamic-goal reward (uses global claimed_by / positions).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from envs.utils.utils import cal_distance


def _active(i: int, robots) -> bool:
    r = robots[i]
    return not r.collision and not r.success


def spatial_clusters(robots, link_dist: float) -> List[List[int]]:
    """Union-find on agents within link_dist (active–active edges only)."""
    n = len(robots)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ld = max(float(link_dist), 1e-6)
    for i in range(n):
        if not _active(i, robots):
            continue
        for j in range(i + 1, n):
            if not _active(j, robots):
                continue
            if cal_distance(robots[i].px, robots[i].py, robots[j].px, robots[j].py) <= ld:
                union(i, j)

    grp: Dict[int, List[int]] = {}
    for i in range(n):
        if not _active(i, robots):
            continue
        r = find(i)
        grp.setdefault(r, []).append(i)
    return list(grp.values())


def build_step_cache(env: Any, for_feature: float) -> Dict[str, Any]:
    args = env.args
    robots = env.robots
    K = int(env.num_goal_targets)
    cb = getattr(env, "claimed_by", None) or []
    unclaimed: List[int] = []
    for k in range(K):
        if k < len(cb) and int(cb[k]) < 0:
            unclaimed.append(k)

    link = float(getattr(args, "dynamic_spatial_cluster_link_dist", 4.0))
    clusters = spatial_clusters(robots, link)

    R_near = float(getattr(args, "dynamic_cluster_target_neighborhood_radius", 10.0))
    quota_match = int(getattr(args, "dynamic_cluster_unclaimed_quota_match_agents", 1)) != 0
    min_quota = max(1, int(getattr(args, "dynamic_cluster_unclaimed_near_min_quota", 2)))
    far_frac = max(0.1, min(0.9, float(getattr(args, "dynamic_cluster_far_agent_fraction", 0.35))))

    role: Dict[int, str] = {}
    cluster_of: Dict[int, int] = {}
    members: Dict[int, List[int]] = {}
    cluster_unc_other: Dict[int, List[int]] = {}

    for cid, memb in enumerate(clusters):
        members[cid] = list(memb)
        for i in memb:
            cluster_of[i] = cid
        active_mem = [i for i in memb if _active(i, robots)]
        if not active_mem:
            continue
        sx = sum(float(robots[i].px) for i in active_mem)
        sy = sum(float(robots[i].py) for i in active_mem)
        cx, cy = sx / len(active_mem), sy / len(active_mem)
        unc_near = [
            k
            for k in unclaimed
            if cal_distance(cx, cy, env.goal_positions[k][0], env.goal_positions[k][1]) <= R_near
        ]
        unc_near_set = set(unc_near)
        unc_other = [k for k in unclaimed if k not in unc_near_set]
        cluster_unc_other[cid] = unc_other
        n_unc = len(unc_near)
        quota = len(active_mem) if quota_match else min_quota
        near_ok = n_unc >= quota

        na = len(active_mem)
        n_explore = max(1, int(math.ceil(na * far_frac)))
        if n_explore >= na:
            n_explore = max(1, na - 1)
        explore_set = set()
        if not near_ok:
            if unc_other:
                d_list: List[tuple] = []
                for i in active_mem:
                    ri = robots[i]
                    bd = min(
                        cal_distance(ri.px, ri.py, env.goal_positions[k][0], env.goal_positions[k][1])
                        for k in unc_other
                    )
                    d_list.append((bd, i))
                d_list.sort(key=lambda x: (x[0], x[1]))
                explore_set = {i for _, i in d_list[:n_explore]}
            else:
                dist_pairs: List[tuple] = []
                for i in active_mem:
                    d = getattr(robots[i], "dist_nearest_unclaimed", None)
                    dist_pairs.append((i, float(d) if d is not None else 1e9))
                dist_pairs.sort(key=lambda x: -x[1])
                explore_set = {i for i, _ in dist_pairs[:n_explore]}

        for i in active_mem:
            if near_ok:
                role[i] = "local_unc"
            elif i in explore_set:
                role[i] = "explore"
            else:
                role[i] = "rest"

    for i, robot in enumerate(robots):
        robot.pre_dist_unc_near_cluster = getattr(robot, "dist_unc_near_cluster", None)
        robot.pre_dist_unclaimed_outside_near = getattr(robot, "dist_unclaimed_outside_near", None)
        if not _active(i, robots):
            robot.dist_unc_near_cluster = None
            robot.dist_unclaimed_outside_near = None
            continue
        cid = cluster_of.get(i, -1)
        if cid < 0:
            robot.dist_unc_near_cluster = None
            robot.dist_unclaimed_outside_near = None
            continue
        active_mem = [j for j in members[cid] if _active(j, robots)]
        if not active_mem:
            robot.dist_unc_near_cluster = None
            robot.dist_unclaimed_outside_near = None
            continue
        sx = sum(float(robots[j].px) for j in active_mem)
        sy = sum(float(robots[j].py) for j in active_mem)
        cx, cy = sx / len(active_mem), sy / len(active_mem)
        unc_near = [
            k
            for k in unclaimed
            if cal_distance(cx, cy, env.goal_positions[k][0], env.goal_positions[k][1]) <= R_near
        ]
        if not unc_near:
            robot.dist_unc_near_cluster = None
        else:
            bd = float("inf")
            for k in unc_near:
                gx, gy = env.goal_positions[k]
                d = cal_distance(robot.px, robot.py, gx, gy)
                if d < bd:
                    bd = d
            robot.dist_unc_near_cluster = float(bd)

        uo = cluster_unc_other.get(cid, [])
        if role.get(i) == "explore" and uo:
            bd_o = float("inf")
            for k in uo:
                gx, gy = env.goal_positions[k]
                d = cal_distance(robot.px, robot.py, gx, gy)
                if d < bd_o:
                    bd_o = d
            robot.dist_unclaimed_outside_near = float(bd_o)
        else:
            robot.dist_unclaimed_outside_near = None

    return {
        "for_feature": for_feature,
        "clusters": clusters,
        "role": role,
        "cluster_of": cluster_of,
        "members": members,
        "K": K,
        "unclaimed": unclaimed,
    }
