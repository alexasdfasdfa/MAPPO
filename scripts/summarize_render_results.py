#!/usr/bin/env python3
"""
Aggregate statistics from MAPPO render outputs under results/render/run*/.

Reads succ/success_agent*.txt and coords/coords_agent*.txt produced by EnvRunner.render().

Trajectory metrics (env index 0 only when lines contain " | " multi-env chunks):
  - pair_success_rate, episode_all_agents_success_rate,
  - mean_all_agents_formation_step (first 1-based step where every agent's succ flag is 1; mean over episodes
    that achieve full formation within the horizon; complements Laplacian type-2 when similarity rates are 0),
  - mean_first_success_step (mean of per-agent first success among successful agent-episode pairs), mean_path_length

Episode meta (episode_meta.json preferred, else episode_meta.jsonl), when target_ids_by_step + goal_positions exist:
  - meta_target_mean / meta_target_var: pooled over all steps and agents in target_ids_by_step
  - meta_target_switch_freq: mean per-episode switch rate (switches / (n_agents * (T-1))); ~0 for fixed slot assignment
  - meta_shape_similarity: mean over episodes of opt_assignment_cost / current_assignment_cost (last step)
  - meta_shaping_reference_mean: mean episode sum of scaled shaping channels (if key present in meta)
If no meta file, or rows lack target_ids_by_step / goal_positions: these show "-".

Mode ``laplacian_type2`` (requires ``episode_meta`` with ``agent_goals`` or ``goal_positions`` of shape n×2):
  Rebuilds symmetric normalized graph Laplacians from replay coords (same weight as env: squared distance)
  and compares current formation L_hat to desired L_des from goal positions at episode start.
  **Type-1 success** = reach assigned target (existing succ/*.txt flags).
  **Type-2 success** = Laplacian similarity above threshold (default 0.97), distinct from target arrival.
  Reports: mean count of timesteps with type-2 per episode, mean first timestep (1-based) when type-2 holds,
  and mean agent path length (optionally only for episodes with ≥1 type-2 step).

  Additionally, **S_L** = 1 - ||L_hat - L_des||_F / ||L_des||_F (same L_hat, L_des as above):
  mean fraction of steps per episode with S_L >= ``--sl-threshold`` (``sl_type2_mean_step_rate``),
  mean timesteps / first step / path length under S_L, and ``sl_type2_episode_rate`` (share of episodes
  with at least one S_L-success step).

  The Laplacian-similarity metric (``--laplacian-sim-metric`` / ``--laplacian-threshold``) has its own
  ``lap_type2_mean_step_rate`` and ``lap_type2_episode_rate`` (parallel to the S_L pair).

Performance: Laplacian replay uses batched NumPy (all timesteps per episode in one pass); meta rows are
indexed by episode id; trajectory length uses vectorized segment sums.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


_REPO = _repo_root()
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _scaled_laplacian_batch(xy: np.ndarray) -> np.ndarray | None:
    """
    Vectorized symmetric normalized Laplacian (same as EnvCore.step).
    xy: (T, n, 2) or (n, 2). Returns (T, n, n) or (1, n, n); None if n < 2.
    """
    if xy.ndim == 2:
        xy = xy[np.newaxis, ...]
    _t, n, _ = xy.shape
    if n < 2:
        return None
    diff = xy[:, :, np.newaxis, :] - xy[:, np.newaxis, :, :]
    w = np.sum(diff * diff, axis=-1)
    row_sums = np.sum(w, axis=2)
    eps = 1e-8
    inv_sqrt = np.power(np.where(row_sums > 0, row_sums, eps), -0.5)
    diag_l = row_sums[:, :, np.newaxis] * np.eye(n, dtype=np.float64)
    l_mat = diag_l - w
    inv_i = inv_sqrt[:, :, np.newaxis]
    inv_j = inv_sqrt[:, np.newaxis, :]
    return inv_i * l_mat * inv_j


def load_episode_meta_records(run_dir: Path) -> list[dict[str, Any]] | None:
    """
    Prefer episode_meta.json (JSON array or single object); else episode_meta.jsonl.
    """
    pj = run_dir / "episode_meta.json"
    pl = run_dir / "episode_meta.jsonl"
    if pj.is_file():
        try:
            raw = json.loads(pj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            return [raw]
        return None
    if pl.is_file():
        out: list[dict[str, Any]] = []
        try:
            with pl.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            return None
        return out
    return None


def _target_trace_stats(trace: list[list[int]]) -> tuple[float, float, float]:
    """Pooled mean, population var, switch frequency = switches / (n * (T-1))."""
    if not trace or not trace[0]:
        return float("nan"), float("nan"), float("nan")
    flat = [int(x) for row in trace for x in row]
    a = np.asarray(flat, dtype=np.float64)
    mean = float(a.mean())
    var = float(a.var())
    T, n = len(trace), len(trace[0])
    if T < 2:
        return mean, var, float("nan")
    sw = 0
    for t in range(1, T):
        for j in range(n):
            if int(trace[t][j]) != int(trace[t - 1][j]):
                sw += 1
    return mean, var, float(sw / (n * (T - 1)))


def _shape_similarity_last(
    positions: np.ndarray,
    goal_xy: np.ndarray,
    target_ids: list[int],
) -> float:
    """opt_cost / (assigned_cost + eps), clipped to [0, 1]."""
    from envs.utils.hungarian_opt import assignment_cost_for_target_ids, optimal_assignment_cost

    if positions.size == 0 or goal_xy.size == 0:
        return float("nan")
    tids = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    cur = assignment_cost_for_target_ids(positions, goal_xy, tids)
    try:
        _, opt = optimal_assignment_cost(positions, goal_xy)
    except Exception:
        return float("nan")
    if not math.isfinite(cur) or not math.isfinite(opt):
        return float("nan")
    if cur <= 1e-12:
        return 1.0 if opt <= 1e-12 else 0.0
    return float(min(1.0, max(0.0, opt / cur)))


def _scaled_laplacian_from_positions(xy: np.ndarray) -> np.ndarray | None:
    """
    Match EnvCore.step: W_ij = (xi-xj)^2+(yi-yj)^2, L = D - W, symmetric normalized L_hat.
    xy: (n, 2) agent positions.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        return None
    b = _scaled_laplacian_batch(xy)
    if b is None:
        return None
    return b[0]


def laplacian_similarity(
    L_hat: np.ndarray,
    L_des: np.ndarray,
    *,
    metric: str = "cosine",
) -> float:
    """
    ``cosine``: Frobenius cosine in [-1, 1].
    ``cosine01``: (cosine + 1) / 2 in [0, 1] (e.g. threshold 0.97 ≈ 97% on unit scale).
    ``rel_frob``: 1 - ||L_hat-L_des||_F^2 / (||L_hat||_F^2 + ||L_des||_F^2) (≈1 when match).
    """
    if metric in ("cosine", "cosine01"):
        a = L_hat.reshape(-1)
        b = L_des.reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-14 or nb < 1e-14:
            return float("nan")
        c = float(np.dot(a, b) / (na * nb))
        if metric == "cosine01":
            return float(0.5 * (c + 1.0))
        return c
    if metric == "rel_frob":
        diff = L_hat - L_des
        num = float(np.sum(diff * diff))
        den = float(np.sum(L_hat * L_hat) + np.sum(L_des * L_des)) + 1e-12
        return float(1.0 - num / den)
    raise ValueError(f"unknown metric: {metric}")


def laplacian_similarity_batch(
    L_hat: np.ndarray,
    L_des: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    """L_hat: (T, n, n); L_des: (n, n). Returns shape (T,) float."""
    metric = str(metric)
    t = int(L_hat.shape[0])
    if metric in ("cosine", "cosine01"):
        a = L_hat.reshape(t, -1)
        b = L_des.reshape(-1)
        na = np.linalg.norm(a, axis=1)
        nb = float(np.linalg.norm(b))
        c = np.sum(a * b, axis=1) / (na * nb + 1e-14)
        if metric == "cosine01":
            return np.asarray(0.5 * (c + 1.0), dtype=np.float64)
        return np.asarray(c, dtype=np.float64)
    if metric == "rel_frob":
        diff = L_hat - L_des
        num = np.sum(diff * diff, axis=(1, 2))
        den = np.sum(L_hat * L_hat, axis=(1, 2)) + float(np.sum(L_des * L_des)) + 1e-12
        return 1.0 - num / den
    raise ValueError(f"unknown metric: {metric}")


def laplacian_S_L_batch(L_hat: np.ndarray, L_des: np.ndarray) -> np.ndarray:
    """L_hat: (T, n, n). Returns shape (T,)."""
    diff = L_hat - L_des
    nf = np.sqrt(np.sum(diff * diff, axis=(1, 2)))
    den = float(np.linalg.norm(L_des, ord="fro"))
    if den < 1e-14:
        return np.full(L_hat.shape[0], np.nan, dtype=np.float64)
    return 1.0 - nf / den


def laplacian_S_L(L_hat: np.ndarray, L_des: np.ndarray) -> float:
    """
    S_L = 1 - ||L_hat - L_des||_F / ||L_des||_F (relative Frobenius gap vs desired Laplacian norm).
    Unbounded below when error exceeds ||L_des||_F; equals 1 when L_hat == L_des.
    """
    diff = L_hat - L_des
    nf = float(np.linalg.norm(diff, ord="fro"))
    den = float(np.linalg.norm(L_des, ord="fro"))
    if den < 1e-14:
        return float("nan")
    return float(1.0 - nf / den)


def _targets_xy_from_meta(rec: dict[str, Any], n_agents: int) -> np.ndarray | None:
    """Prefer agent_goals (per-robot gx,gy); else goal_positions if length matches."""
    ag = rec.get("agent_goals")
    if isinstance(ag, list) and len(ag) == n_agents:
        g = np.asarray(ag, dtype=np.float64)
        if g.ndim == 2 and g.shape[1] == 2:
            return g
    gp = rec.get("goal_positions")
    if gp is not None:
        g = np.asarray(gp, dtype=np.float64)
        if g.ndim == 2 and g.shape[1] == 2 and g.shape[0] == n_agents:
            return g
    return None


def episode_positions_T_n_2(
    coords_by_agent: dict[int, dict[int, list[tuple[float, float]]]],
    agent_ids: list[int],
    ep: int,
) -> np.ndarray | None:
    """Stack coords to (T, n, 2); same length across agents."""
    seqs: list[list[tuple[float, float]]] = []
    for aid in agent_ids:
        s = coords_by_agent[aid].get(ep, [])
        if not s:
            return None
        seqs.append(s)
    t_max = min(len(s) for s in seqs)
    if t_max < 1:
        return None
    cols = [np.asarray(s[:t_max], dtype=np.float64) for s in seqs]
    return np.stack(cols, axis=1)


def meta_by_episode_map(meta_records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for r in meta_records:
        try:
            epi = int(r.get("episode", -1))
        except (TypeError, ValueError):
            continue
        if epi >= 0:
            out[epi] = r
    return out


def meta_record_for_episode(meta_records: list[dict[str, Any]], ep: int) -> dict[str, Any] | None:
    for r in meta_records:
        if int(r.get("episode", -1)) == int(ep):
            return r
    return None


def analyze_laplacian_type2_stats(
    episodes: list[int],
    coords_by_agent: dict[int, dict[int, list[tuple[float, float]]]],
    agent_ids: list[int],
    meta_records: list[dict[str, Any]] | None,
    *,
    sim_threshold: float,
    sim_metric: str,
    sl_threshold: float,
) -> dict[str, Any]:
    """
    Type-2 success: Laplacian similarity above threshold (vs type-1 = reach goal in succ files).
    Uses goal positions at episode start from meta (dynamic goal motion between steps is not replayed).

    S_L row uses ``sl_threshold`` on :func:`laplacian_S_L` (independent of ``sim_metric``).
    """
    dash = {
        "lap_type2_threshold": "-",
        "lap_type2_sim_metric": "-",
        "lap_type2_mean_timesteps": "-",
        "lap_type2_mean_first_step": "-",
        "lap_type2_mean_path_length": "-",
        "lap_type2_mean_step_rate": "-",
        "lap_type2_episode_rate": "-",
        "lap_type2_episodes_used": "-",
        "lap_type2_episodes_skipped": "-",
        "sl_S_L_threshold": "-",
        "sl_type2_mean_timesteps": "-",
        "sl_type2_mean_first_step": "-",
        "sl_type2_mean_path_length": "-",
        "sl_type2_mean_step_rate": "-",
        "sl_type2_episode_rate": "-",
    }
    if not meta_records:
        return dash
    n_agents = len(agent_ids)
    thr = float(sim_threshold)
    metric = str(sim_metric)
    sl_thr = float(sl_threshold)

    step_counts: list[int] = []
    first_steps: list[int] = []
    step_counts_sl: list[int] = []
    first_steps_sl: list[int] = []
    step_rates_lap: list[float] = []
    ep_lap_hit: list[int] = []
    step_rates_sl: list[float] = []
    ep_sl_hit: list[int] = []
    path_sum = 0.0
    path_n = 0
    used = 0
    skipped = 0

    for ep in episodes:
        rec = meta_record_for_episode(meta_records, ep)
        if rec is None:
            skipped += 1
            continue
        tgt = _targets_xy_from_meta(rec, n_agents)
        if tgt is None:
            skipped += 1
            continue
        L_des = _scaled_laplacian_from_positions(tgt)
        if L_des is None:
            skipped += 1
            continue

        pos = episode_positions_T_n_2(coords_by_agent, agent_ids, ep)
        if pos is None:
            skipped += 1
            continue

        l_hat_all = _scaled_laplacian_batch(pos)
        if l_hat_all is None:
            skipped += 1
            continue
        sim_vec = laplacian_similarity_batch(l_hat_all, L_des, metric=metric)
        sl_vec = laplacian_S_L_batch(l_hat_all, L_des)
        T = int(pos.shape[0])

        ok_sim = np.isfinite(sim_vec) & (sim_vec >= thr)
        ge_ts = int(np.sum(ok_sim))
        if ok_sim.any():
            first_s = int(np.argmax(ok_sim) + 1)
        else:
            first_s = None

        ok_sl = np.isfinite(sl_vec) & (sl_vec >= sl_thr)
        ge_ts_sl = int(np.sum(ok_sl))
        if ok_sl.any():
            first_s_sl = int(np.argmax(ok_sl) + 1)
        else:
            first_s_sl = None

        step_counts.append(ge_ts)
        if first_s is not None:
            first_steps.append(first_s)
        step_rates_lap.append(float(ge_ts) / float(max(T, 1)))
        ep_lap_hit.append(1 if ge_ts > 0 else 0)
        step_counts_sl.append(ge_ts_sl)
        if first_s_sl is not None:
            first_steps_sl.append(first_s_sl)
        step_rates_sl.append(float(ge_ts_sl) / float(max(T, 1)))
        ep_sl_hit.append(1 if ge_ts_sl > 0 else 0)

        for aid in agent_ids:
            pts = coords_by_agent[aid].get(ep, [])
            path_sum += path_length(pts)
            path_n += 1

        used += 1

    if used == 0:
        out = dict(dash)
        out["lap_type2_threshold"] = thr
        out["lap_type2_sim_metric"] = metric
        out["sl_S_L_threshold"] = sl_thr
        out["lap_type2_episodes_skipped"] = skipped
        return out

    mean_first = float(np.mean(first_steps)) if first_steps else float("nan")
    mean_steps = float(np.mean(step_counts)) if step_counts else float("nan")
    mean_path = float(path_sum / path_n) if path_n else float("nan")
    mean_step_rate_lap = float(np.mean(step_rates_lap)) if step_rates_lap else float("nan")
    ep_rate_lap = float(np.mean(ep_lap_hit)) if ep_lap_hit else float("nan")

    mean_first_sl = float(np.mean(first_steps_sl)) if first_steps_sl else float("nan")
    mean_steps_sl = float(np.mean(step_counts_sl)) if step_counts_sl else float("nan")
    mean_step_rate_sl = float(np.mean(step_rates_sl)) if step_rates_sl else float("nan")
    ep_rate_sl = float(np.mean(ep_sl_hit)) if ep_sl_hit else float("nan")

    return {
        "lap_type2_threshold": thr,
        "lap_type2_sim_metric": metric,
        "lap_type2_mean_timesteps": mean_steps,
        "lap_type2_mean_first_step": mean_first,
        "lap_type2_mean_path_length": mean_path,
        "lap_type2_mean_step_rate": mean_step_rate_lap,
        "lap_type2_episode_rate": ep_rate_lap,
        "lap_type2_episodes_used": used,
        "lap_type2_episodes_skipped": skipped,
        "sl_S_L_threshold": sl_thr,
        "sl_type2_mean_timesteps": mean_steps_sl,
        "sl_type2_mean_first_step": mean_first_sl,
        "sl_type2_mean_path_length": mean_path,
        "sl_type2_mean_step_rate": mean_step_rate_sl,
        "sl_type2_episode_rate": ep_rate_sl,
    }


def _final_positions_matrix(
    coords_by_agent: dict[int, dict[int, list[tuple[float, float]]]],
    agent_ids: list[int],
    episode: int,
) -> np.ndarray | None:
    rows = []
    for aid in agent_ids:
        seq = coords_by_agent[aid].get(episode, [])
        if not seq:
            return None
        rows.append([seq[-1][0], seq[-1][1]])
    return np.asarray(rows, dtype=np.float64)


def analyze_episode_meta_effect(
    run_dir: Path,
    meta_records: list[dict[str, Any]] | None,
    agent_ids: list[int],
    coords_by_agent: dict[int, dict[int, list[tuple[float, float]]]],
) -> dict[str, Any]:
    """
    Four metrics when meta has target_ids_by_step and goal_positions (dynamic, undetermined, or static slot).
    Otherwise string '-'.
    """
    dash = {
        "meta_target_mean": "-",
        "meta_target_var": "-",
        "meta_target_switch_freq": "-",
        "meta_shape_similarity": "-",
        "meta_shaping_reference_mean": "-",
    }
    if not meta_records:
        return dash

    usable = [
        r
        for r in meta_records
        if r.get("target_ids_by_step")
        and r.get("goal_positions")
        and isinstance(r["target_ids_by_step"], list)
    ]
    if not usable:
        return dash

    all_ids: list[int] = []
    switch_freqs: list[float] = []
    shape_sims: list[float] = []
    shapings: list[float] = []

    for r in usable:
        tr = r["target_ids_by_step"]
        if not tr:
            continue
        m, v, sf = _target_trace_stats(tr)
        if math.isfinite(m):
            all_ids.extend(int(x) for row in tr for x in row)
        if math.isfinite(sf):
            switch_freqs.append(sf)

        ep = int(r.get("episode", -1))
        gp = np.asarray(r["goal_positions"], dtype=np.float64)
        if ep >= 0 and gp.ndim == 2 and gp.shape[0] > 0:
            P = _final_positions_matrix(coords_by_agent, agent_ids, ep)
            last_tids = [int(x) for x in tr[-1]]
            if P is not None and P.shape[0] == len(last_tids):
                sim = _shape_similarity_last(P, gp, last_tids)
                if math.isfinite(sim):
                    shape_sims.append(sim)

        if "shaping_reward_reference" in r:
            try:
                shapings.append(float(r["shaping_reward_reference"]))
            except (TypeError, ValueError):
                pass

    if not all_ids:
        out = dict(dash)
        if shapings:
            out["meta_shaping_reference_mean"] = float(np.mean(shapings))
        return out

    arr = np.asarray(all_ids, dtype=np.float64)
    out = {
        "meta_target_mean": float(arr.mean()),
        "meta_target_var": float(arr.var()),
        "meta_target_switch_freq": float(np.mean(switch_freqs)) if switch_freqs else "-",
        "meta_shape_similarity": float(np.mean(shape_sims)) if shape_sims else "-",
        "meta_shaping_reference_mean": float(np.mean(shapings)) if shapings else "-",
    }
    return out


def parse_run_headers(succ_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    p0 = succ_dir / "success_agent0.txt"
    if not p0.is_file():
        return meta
    with p0.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("#"):
                break
            if "agents=" in s:
                m = re.search(r"agents=(\d+)", s)
                if m:
                    meta["num_agents"] = int(m.group(1))
            if "rollout_length=" in s:
                m = re.search(r"rollout_length=(\d+)", s)
                if m:
                    meta["rollout_length"] = int(m.group(1))
            m = re.search(r"pattern=([^,\s#]+)", s)
            if m:
                meta["pattern"] = m.group(1).strip()
    return meta


def _split_env_chunks(rest: str) -> list[str]:
    return [c.strip() for c in rest.split("|")]


def parse_succ_data_line(line: str) -> tuple[int, list[list[int]]] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "," not in line:
        return None
    ep_str, rest = line.split(",", 1)
    episode = int(ep_str.strip())
    chunks: list[list[int]] = []
    for ch in _split_env_chunks(rest):
        bits = [int(x.strip()) for x in ch.split(";") if x.strip()]
        chunks.append(bits)
    return episode, chunks


def parse_coords_data_line(line: str) -> tuple[int, list[list[tuple[float, float]]]] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "," not in line:
        return None
    ep_str, rest = line.split(",", 1)
    episode = int(ep_str.strip())
    chunks: list[list[tuple[float, float]]] = []
    for ch in _split_env_chunks(rest):
        pairs: list[tuple[float, float]] = []
        for seg in ch.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            parts = seg.split()
            if len(parts) >= 2:
                pairs.append((float(parts[0]), float(parts[1])))
        chunks.append(pairs)
    return episode, chunks


def path_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    d = np.diff(arr, axis=0)
    return float(np.sqrt(np.sum(d * d, axis=1)).sum())


def first_success_step_1based(flags: list[int]) -> int | None:
    for i, v in enumerate(flags):
        if v:
            return i + 1
    return None


def first_all_agents_formation_step_1based(
    succ_by_agent: dict[int, dict[int, list[int]]],
    agent_ids: list[int],
    episode: int,
) -> int | None:
    """
    Earliest 1-based timestep t such that every agent has succ flag 1 at step t-1.
    Render logs mark 1 once reach_goal holds and leave it 1, so this equals max_i first_success_step(agent i)
    when full formation is reached; None if trajectories missing or no step has all agents at goal.
    """
    series: list[list[int]] = []
    for aid in agent_ids:
        fl = succ_by_agent[aid].get(episode, [])
        if not fl:
            return None
        series.append(fl)
    t_max = min(len(s) for s in series)
    if t_max < 1:
        return None
    for t in range(t_max):
        if all(int(s[t]) != 0 for s in series):
            return t + 1
    return None


def load_agent_succ_episodes(succ_dir: Path, agent_id: int) -> dict[int, list[int]]:
    p = succ_dir / f"success_agent{agent_id}.txt"
    out: dict[int, list[int]] = {}
    if not p.is_file():
        return out
    with p.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_succ_data_line(line)
            if parsed is None:
                continue
            ep, chunks = parsed
            if not chunks:
                continue
            out[ep] = chunks[0]
    return out


def load_agent_coords_episodes(coords_dir: Path, agent_id: int) -> dict[int, list[tuple[float, float]]]:
    p = coords_dir / f"coords_agent{agent_id}.txt"
    out: dict[int, list[tuple[float, float]]] = {}
    if not p.is_file():
        return out
    with p.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_coords_data_line(line)
            if parsed is None:
                continue
            ep, chunks = parsed
            if not chunks:
                continue
            out[ep] = chunks[0]
    return out


def _lap_type2_placeholder_row() -> dict[str, Any]:
    return {
        "lap_type2_threshold": "-",
        "lap_type2_sim_metric": "-",
        "lap_type2_mean_timesteps": "-",
        "lap_type2_mean_first_step": "-",
        "lap_type2_mean_path_length": "-",
        "lap_type2_mean_step_rate": "-",
        "lap_type2_episode_rate": "-",
        "lap_type2_episodes_used": "-",
        "lap_type2_episodes_skipped": "-",
        "sl_S_L_threshold": "-",
        "sl_type2_mean_timesteps": "-",
        "sl_type2_mean_first_step": "-",
        "sl_type2_mean_path_length": "-",
        "sl_type2_mean_step_rate": "-",
        "sl_type2_episode_rate": "-",
    }


def analyze_run(
    run_dir: Path,
    *,
    mode: str = "default",
    laplacian_threshold: float = 0.97,
    laplacian_sim_metric: str = "cosine01",
    sl_threshold: float = 0.97,
) -> dict[str, Any] | None:
    succ_dir = run_dir / "succ"
    coords_dir = run_dir / "coords"
    if not succ_dir.is_dir() or not coords_dir.is_dir():
        return None
    meta = parse_run_headers(succ_dir)
    agent_files = sorted(succ_dir.glob("success_agent*.txt"))
    if not agent_files:
        return None
    agent_ids: list[int] = []
    for fp in agent_files:
        m = re.search(r"success_agent(\d+)\.txt$", fp.name)
        if m:
            agent_ids.append(int(m.group(1)))
    agent_ids.sort()
    n_agents = len(agent_ids)

    # Load all (episode -> flags / points)
    succ_by_agent: dict[int, dict[int, list[int]]] = {}
    coords_by_agent: dict[int, dict[int, list[tuple[float, float]]]] = {}
    for aid in agent_ids:
        succ_by_agent[aid] = load_agent_succ_episodes(succ_dir, aid)
        coords_by_agent[aid] = load_agent_coords_episodes(coords_dir, aid)

    all_eps: set[int] = set()
    for d in succ_by_agent.values():
        all_eps.update(d.keys())
    for d in coords_by_agent.values():
        all_eps.update(d.keys())
    episodes = sorted(all_eps)
    if not episodes:
        return None

    pair_total = 0
    pair_ok = 0
    first_steps: list[int] = []
    formation_steps: list[int] = []
    path_lengths: list[float] = []
    episodes_all_ok = 0

    for ep in episodes:
        ep_all_success = True
        fs_form = first_all_agents_formation_step_1based(succ_by_agent, agent_ids, ep)
        if fs_form is not None:
            formation_steps.append(fs_form)
        for aid in agent_ids:
            pair_total += 1
            flags = succ_by_agent[aid].get(ep, [])
            pts = coords_by_agent[aid].get(ep, [])
            if flags and max(flags):
                pair_ok += 1
                fs = first_success_step_1based(flags)
                if fs is not None:
                    first_steps.append(fs)
            else:
                ep_all_success = False
            if len(pts) >= 2:
                path_lengths.append(path_length(pts))
            else:
                path_lengths.append(0.0)
        if ep_all_success:
            episodes_all_ok += 1

    rl = meta.get("rollout_length")
    if rl is None and episodes:
        # infer from first available trajectory
        for aid in agent_ids:
            fl = succ_by_agent[aid].get(episodes[0], [])
            if fl:
                rl = len(fl)
                break

    meta_recs = load_episode_meta_records(run_dir)
    effect = analyze_episode_meta_effect(run_dir, meta_recs, agent_ids, coords_by_agent)

    row = {
        "run": run_dir.name,
        "num_agents": meta.get("num_agents", n_agents),
        "num_episodes": len(episodes),
        "rollout_length": rl,
        "pattern": meta.get("pattern", ""),
        "pair_success_rate": pair_ok / pair_total if pair_total else 0.0,
        "episode_all_agents_success_rate": episodes_all_ok / len(episodes) if episodes else 0.0,
        "mean_all_agents_formation_step": (
            float(sum(formation_steps) / len(formation_steps)) if formation_steps else float("nan")
        ),
        "formation_complete_episode_rate": (
            float(len(formation_steps) / len(episodes)) if episodes else float("nan")
        ),
        "mean_first_success_step": sum(first_steps) / len(first_steps) if first_steps else float("nan"),
        "mean_path_length": sum(path_lengths) / len(path_lengths) if path_lengths else float("nan"),
        "num_success_pairs": pair_ok,
        "num_pairs": pair_total,
    }
    row.update(effect)

    if mode == "laplacian_type2":
        row.update(
            analyze_laplacian_type2_stats(
                episodes,
                coords_by_agent,
                agent_ids,
                meta_recs,
                sim_threshold=laplacian_threshold,
                sim_metric=laplacian_sim_metric,
                sl_threshold=sl_threshold,
            )
        )
    else:
        row.update(_lap_type2_placeholder_row())
    return row


def iter_run_dirs(render_root: Path) -> list[Path]:
    if not render_root.is_dir():
        return []
    runs = []
    for p in render_root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"run(\d+)$", p.name, re.I)
        if m:
            runs.append((int(m.group(1)), p))
    runs.sort(key=lambda x: x[0])
    return [p for _, p in runs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Path to results/render (default: <repo>/results/render)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write one CSV row per run to this path",
    )
    ap.add_argument(
        "--mode",
        choices=("default", "laplacian_type2"),
        default="default",
        help="laplacian_type2: extra stats for type-2 success (Laplacian similarity vs goal formation, "
        "distinct from target-arrival / succ flags).",
    )
    ap.add_argument(
        "--laplacian-threshold",
        type=float,
        default=0.97,
        help="Type-2 success when similarity >= this value (with --mode laplacian_type2).",
    )
    ap.add_argument(
        "--laplacian-sim-metric",
        choices=("cosine01", "cosine", "rel_frob"),
        default="cosine01",
        help="cosine01: (cos+1)/2 in [0,1] for threshold like 0.97; cosine: raw [-1,1]; rel_frob: 1-rel error.",
    )
    ap.add_argument(
        "--sl-threshold",
        type=float,
        default=0.97,
        help="Similarity success using S_L=1-||L_hat-L_des||_F/||L_des||_F: count steps with S_L >= this value.",
    )
    args = ap.parse_args()
    root = args.root
    if root is None:
        root = _repo_root() / "results" / "render"
    root = root.resolve()

    rows: list[dict[str, Any]] = []
    for run_dir in iter_run_dirs(root):
        row = analyze_run(
            run_dir,
            mode=str(args.mode),
            laplacian_threshold=float(args.laplacian_threshold),
            laplacian_sim_metric=str(args.laplacian_sim_metric),
            sl_threshold=float(args.sl_threshold),
        )
        if row:
            rows.append(row)

    if not rows:
        print(f"No render runs found under {root} (expected run*/succ, run*/coords).", file=sys.stderr)
        return 1

    headers = [
        "run",
        "num_agents",
        "num_episodes",
        "rollout_length",
        "pattern",
        "pair_success_rate",
        "episode_all_agents_success_rate",
        "mean_all_agents_formation_step",
        "formation_complete_episode_rate",
        "mean_first_success_step",
        "mean_path_length",
        "num_success_pairs",
        "num_pairs",
        "meta_target_mean",
        "meta_target_var",
        "meta_target_switch_freq",
        "meta_shape_similarity",
        "meta_shaping_reference_mean",
        "lap_type2_threshold",
        "lap_type2_sim_metric",
        "lap_type2_mean_timesteps",
        "lap_type2_mean_first_step",
        "lap_type2_mean_path_length",
        "lap_type2_mean_step_rate",
        "lap_type2_episode_rate",
        "lap_type2_episodes_used",
        "lap_type2_episodes_skipped",
        "sl_S_L_threshold",
        "sl_type2_mean_timesteps",
        "sl_type2_mean_first_step",
        "sl_type2_mean_path_length",
        "sl_type2_mean_step_rate",
        "sl_type2_episode_rate",
    ]

    def fmt(x: Any) -> str:
        if x == "-":
            return "-"
        if isinstance(x, float):
            if math.isnan(x):
                return "nan"
            return f"{x:.6f}".rstrip("0").rstrip(".")
        return str(x)

    colw = max(len(h) for h in headers)
    for row in rows:
        print(f"=== {row['run']} ===")
        for h in headers:
            print(f"  {h:<{colw}}  {fmt(row[h])}")
        print()

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for row in rows:
                out = {}
                for k in headers:
                    v = row[k]
                    if isinstance(v, float) and math.isnan(v):
                        out[k] = ""
                    else:
                        out[k] = v
                w.writerow(out)
        print(f"Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
