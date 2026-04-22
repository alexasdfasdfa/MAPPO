#!/usr/bin/env python3
"""
Build one multi-agent trajectory GIF per episode from render coord logs.

Reads:  <repo>/results/render/run{n}/coords/coords_agent*.txt,
        run{n}/episode_meta.jsonl (pattern, goals; dynamic runs add goal_positions + target_ids_by_step)
Writes: <cwd>/fig/render/n/{ep_id}.gif
        Copies results/render/run{n}/run_flags.txt → <cwd>/fig/render/n/run_flags.txt when present (model / train run id).
        Dynamic: colors follow current target_id (shared palette over K slots); rings match that target.
        Static: ring at goal slot k uses palette[k]; agent i uses palette[i % K]. Coords files sorted by agent id (not lexicographic).

Example:
  python visualize_render_trajectories.py 13
  python visualize_render_trajectories.py 13 --episode 3 --stride 4   # fewer frames, faster
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def target_slot_colors(num_slots: int) -> np.ndarray:
    """
    RGBA (K, 4): one deterministic color per target slot id 0..K-1.
    Dynamic: trajectory/dot/ring use color[target_ids_by_step[t,i]].
    Static: ring at slot k uses color[k]; agent i uses color[i % K]. No meta: K=A, same as i % K.
    """
    k = max(int(num_slots), 0)
    if k == 0:
        return np.zeros((0, 4), dtype=np.float64)
    cmap = getattr(plt.cm, "turbo", None) or plt.cm.hsv
    return cmap(np.linspace(0.0, 0.95, k, endpoint=True))


def _coords_files_in_agent_order(coords_dir: Path) -> list[Path]:
    """Numeric order: 0,1,2,...,10,... not lexicographic (0,1,10,2,...)."""
    pat = re.compile(r"coords_agent(\d+)\.txt$")
    found: list[tuple[int, Path]] = []
    for p in coords_dir.glob("coords_agent*.txt"):
        m = pat.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found]


def _parse_meta_header(line: str) -> int | None:
    m = re.search(r"agents=(\d+)", line)
    return int(m.group(1)) if m else None


def _parse_coords_file_headers(coords_dir: Path) -> dict:
    """Fallback labels from coords_agent0.txt comment lines (no goals)."""
    fp = coords_dir / "coords_agent0.txt"
    out: dict = {}
    if not fp.is_file():
        return out
    with open(fp, encoding="utf-8") as f:
        lines = []
        for _ in range(3):
            ln = f.readline()
            if not ln:
                break
            lines.append(ln)
    if lines:
        m = re.search(r"pattern=([^,\s#]+)", lines[0])
        if m:
            out["pattern"] = m.group(1).strip()
    if len(lines) > 1 and lines[1].lstrip().startswith("#"):
        m = re.search(r"rollout_length=(\d+)", lines[1])
        if m:
            out["episode_length"] = int(m.group(1))
        m2 = re.search(r"pattern_template_len=(\d+)", lines[1])
        if m2:
            out["pattern_template_len"] = int(m2.group(1))
    return out


def load_episode_meta_jsonl(run_dir: Path) -> dict[int, dict]:
    p = run_dir / "episode_meta.jsonl"
    by_ep: dict[int, dict] = {}
    if not p.is_file():
        return by_ep
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                by_ep[int(o["episode"])] = o
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return by_ep


def _parse_episode_line(line: str) -> tuple[int, np.ndarray] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "," not in line:
        return None
    ep_str, rest = line.split(",", 1)
    try:
        ep_id = int(ep_str.strip())
    except ValueError:
        return None
    rest = rest.strip()
    env_chunks = [c.strip() for c in rest.split("|")]
    chunk0 = env_chunks[0]
    pairs = [p.strip() for p in chunk0.split(";") if p.strip()]
    pts = []
    for p in pairs:
        parts = p.split()
        if len(parts) < 2:
            continue
        pts.append([float(parts[0]), float(parts[1])])
    if not pts:
        return None
    return ep_id, np.asarray(pts, dtype=np.float64)


def load_agent_trajectories(coords_dir: Path) -> tuple[int | None, list[tuple[int, np.ndarray]]]:
    files = _coords_files_in_agent_order(coords_dir)
    if not files:
        raise FileNotFoundError(f"No coords_agent*.txt under {coords_dir}")

    num_agents_meta: int | None = None
    per_agent_episodes: list[list[tuple[int, np.ndarray]]] = []

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            continue
        if num_agents_meta is None:
            num_agents_meta = _parse_meta_header(lines[0])
        eps: list[tuple[int, np.ndarray]] = []
        for ln in lines[1:]:
            parsed = _parse_episode_line(ln)
            if parsed is not None:
                eps.append(parsed)
        per_agent_episodes.append(eps)

    n_files = len(per_agent_episodes)
    if num_agents_meta is not None and n_files != num_agents_meta:
        pass  # still use file count as source of truth

    merged: list[tuple[int, np.ndarray]] = []
    if not per_agent_episodes:
        return num_agents_meta, merged

    ref_eps = {eid for eid, _ in per_agent_episodes[0]}
    for agent_eps in per_agent_episodes[1:]:
        ref_eps &= {eid for eid, _ in agent_eps}

    for eid in sorted(ref_eps):
        trajs = []
        for agent_eps in per_agent_episodes:
            d = dict(agent_eps)
            if eid not in d:
                break
            trajs.append(d[eid])
        if len(trajs) != len(per_agent_episodes):
            continue
        T = min(t.shape[0] for t in trajs)
        stacked = np.stack([t[:T] for t in trajs], axis=1)
        merged.append((eid, stacked))

    return num_agents_meta, merged


def _pick_episode(
    episodes: list[tuple[int, np.ndarray]], episode_id: int | None
) -> tuple[int, np.ndarray]:
    if not episodes:
        raise ValueError("No episode data found in coords files.")
    if episode_id is None:
        return episodes[0]
    for eid, arr in episodes:
        if eid == episode_id:
            return eid, arr
    raise ValueError(f"Episode id {episode_id} not found. Available: {[e[0] for e in episodes]}")


def _axis_bounds(
    traj_all: np.ndarray,
    margin_ratio: float = 0.05,
    extra_xy: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    parts = [traj_all.reshape(-1, 2)]
    if extra_xy is not None and extra_xy.size >= 2:
        parts.append(np.asarray(extra_xy, dtype=np.float64).reshape(-1, 2))
    flat = np.vstack(parts)
    xmin, ymin = flat.min(axis=0)
    xmax, ymax = flat.max(axis=0)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    mx = dx * margin_ratio
    my = dy * margin_ratio
    return xmin - mx, xmax + mx, ymin - my, ymax + my


def render_all_agents_gif(
    traj: np.ndarray,
    bounds: tuple[float, float, float, float],
    out_path: Path,
    episode_id: int,
    stride: int,
    dpi: int,
    duration: float,
    episode_meta: dict | None,
    header_fallback: dict | None,
    target_colors: np.ndarray,
    K: int,
    use_dynamic_goals: bool,
    goal_positions: np.ndarray | None = None,
    target_ids_by_step: np.ndarray | None = None,
    static_goals: list | None = None,
) -> None:
    """
    traj: (T, A, 2).
    target_colors: (K, 4) RGBA — color for target slot id in [0, K).
    Dynamic: each frame agent i uses target_colors[target_ids_by_step[t, i] % K]; ring follows same tid.
    Static: ring at static_goals[k] uses target_colors[k]; agent i uses target_colors[i % K].
    """
    T, A, _ = traj.shape
    indices = list(range(0, T, max(stride, 1)))
    if indices[-1] != T - 1:
        indices.append(T - 1)

    xmin, xmax, ymin, ymax = bounds
    frames: list[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    # Fixed margins; leave room for suptitle when showing pattern / lengths.
    fig.subplots_adjust(left=0.1, right=0.95, top=0.86, bottom=0.1)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    label_src = episode_meta if episode_meta else (header_fallback or {})
    pat = label_src.get("pattern", "?")
    rlen = label_src.get("episode_length", T)
    tpl_n = label_src.get("pattern_template_len", "?")
    fig.suptitle(
        f"pattern={pat}  |  rollout_length={rlen}  |  template_pts={tpl_n}",
        fontsize=10,
        y=0.98,
    )

    K = max(int(K), 1)
    if target_colors.shape[0] < K:
        raise ValueError(f"target_colors rows {target_colors.shape[0]} < K={K}")

    goal_scats: list = []
    if use_dynamic_goals:
        assert goal_positions is not None and target_ids_by_step is not None
        for i in range(A):
            tid0 = int(target_ids_by_step[0, i]) % K
            gx, gy = float(goal_positions[tid0, 0]), float(goal_positions[tid0, 1])
            hex_t = mcolors.to_hex(target_colors[tid0])
            gsc = ax.scatter(
                [gx],
                [gy],
                s=130,
                facecolors="none",
                edgecolors=hex_t,
                linewidths=2.0,
                zorder=4,
            )
            goal_scats.append(gsc)
    elif static_goals:
        for k, g in enumerate(static_goals):
            if len(g) < 2:
                continue
            kk = k % K
            hex_c = mcolors.to_hex(target_colors[kk])
            ax.scatter(
                [float(g[0])],
                [float(g[1])],
                s=130,
                facecolors="none",
                edgecolors=hex_c,
                linewidths=2.0,
                zorder=4,
            )

    lines = []
    scats = []
    for i in range(A):
        tid_init = (
            int(target_ids_by_step[0, i]) % K
            if use_dynamic_goals and target_ids_by_step is not None
            else (i % K)
        )
        hex_c = mcolors.to_hex(target_colors[tid_init])
        (ln,) = ax.plot([], [], color=hex_c, linewidth=1.4, alpha=0.88)
        sc = ax.scatter(
            [],
            [],
            c=[hex_c],
            s=40,
            zorder=5,
            edgecolors="k",
            linewidths=0.4,
        )
        lines.append(ln)
        scats.append(sc)

    title_artist = ax.set_title("")

    for t in indices:
        for i in range(A):
            seg = traj[: t + 1, i, :]
            lines[i].set_data(seg[:, 0], seg[:, 1])
            scats[i].set_offsets(seg[-1:])
            if use_dynamic_goals and target_ids_by_step is not None and goal_positions is not None:
                tid = int(target_ids_by_step[t, i]) % K
                hex_t = mcolors.to_hex(target_colors[tid])
                lines[i].set_color(hex_t)
                scats[i].set_facecolors([hex_t])
            else:
                hex_s = mcolors.to_hex(target_colors[i % K])
                lines[i].set_color(hex_s)
                scats[i].set_facecolors([hex_s])
        if use_dynamic_goals and goal_scats:
            assert goal_positions is not None and target_ids_by_step is not None
            for i in range(A):
                tid = int(target_ids_by_step[t, i]) % K
                gx, gy = float(goal_positions[tid, 0]), float(goal_positions[tid, 1])
                goal_scats[i].set_offsets(np.array([[gx, gy]]))
                goal_scats[i].set_edgecolors([mcolors.to_hex(target_colors[tid])])
        title_artist.set_text(
            f"episode {episode_id}  |  steps 0–{t} / {T - 1}  |  agents 0..{A - 1}"
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())

    plt.close(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, duration=duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory GIFs from render coord logs.")
    parser.add_argument(
        "n",
        type=int,
        help="Render run index: load results/render/run{n}/coords/",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="If set, only generate GIF for this episode id; default: all episodes.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every k-th timestep as a frame (fewer frames → faster render, smaller GIF).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure DPI for each frame.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.04,
        help="Seconds per frame in the GIF.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override results root (default: <repo>/results/render).",
    )
    args = parser.parse_args()

    root = Path(args.data_root) if args.data_root else _repo_root() / "results" / "render"
    coords_dir = root / f"run{args.n}" / "coords"
    if not coords_dir.is_dir():
        raise SystemExit(f"Missing coords directory: {coords_dir}")

    _, episodes = load_agent_trajectories(coords_dir)
    if not episodes:
        raise SystemExit("No merged episode data in coords files.")

    run_dir = coords_dir.parent
    meta_by_ep = load_episode_meta_jsonl(run_dir)
    header_fallback = _parse_coords_file_headers(coords_dir)

    if args.episode is not None:
        to_run = [_pick_episode(episodes, args.episode)]
    else:
        to_run = episodes

    out_dir = Path.cwd() / "fig" / "render" / str(args.n)
    out_dir.mkdir(parents=True, exist_ok=True)
    _flags_src = run_dir / "run_flags.txt"
    if _flags_src.is_file():
        shutil.copy2(_flags_src, out_dir / "run_flags.txt")
        print(f"Copied {_flags_src} -> {out_dir / 'run_flags.txt'}")
        _repo_fig = _repo_root() / "fig" / "render" / str(args.n)
        if _repo_fig.resolve() != out_dir.resolve():
            _repo_fig.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_flags_src, _repo_fig / "run_flags.txt")
            print(f"Copied {_flags_src} -> {_repo_fig / 'run_flags.txt'}")

    for ep_id, traj in to_run:
        T, A, _ = traj.shape
        ep_meta = meta_by_ep.get(ep_id)
        bounds_extras: list[np.ndarray] = []
        if ep_meta and isinstance(ep_meta.get("agent_goals"), list):
            bounds_extras.append(
                np.asarray(ep_meta["agent_goals"], dtype=np.float64).reshape(-1, 2)
            )

        goal_positions = None
        target_ids_by_step = None
        use_dynamic = False
        static_goals: list | None = None
        if (
            ep_meta
            and ep_meta.get("dynamic_target")
            and isinstance(ep_meta.get("goal_positions"), list)
            and isinstance(ep_meta.get("target_ids_by_step"), list)
        ):
            gp = np.asarray(ep_meta["goal_positions"], dtype=np.float64)
            tid = np.asarray(ep_meta["target_ids_by_step"], dtype=np.int64)
            if gp.ndim == 2 and gp.shape[1] == 2 and tid.shape == (T, A):
                goal_positions = gp
                target_ids_by_step = tid
                use_dynamic = True
                bounds_extras.append(gp)

        if use_dynamic:
            K = int(goal_positions.shape[0])
            target_colors = target_slot_colors(K)
        elif ep_meta and isinstance(ep_meta.get("agent_goals"), list) and ep_meta["agent_goals"]:
            static_goals = ep_meta["agent_goals"]
            K = len(static_goals)
            target_colors = target_slot_colors(K)
        else:
            K = max(A, 1)
            target_colors = target_slot_colors(K)
            static_goals = None

        extra_xy = np.vstack(bounds_extras) if bounds_extras else None
        bounds = _axis_bounds(traj, extra_xy=extra_xy)
        out_path = out_dir / f"{ep_id}.gif"
        render_all_agents_gif(
            traj,
            bounds,
            out_path,
            episode_id=ep_id,
            stride=args.stride,
            dpi=args.dpi,
            duration=args.duration,
            episode_meta=ep_meta,
            header_fallback=header_fallback if ep_meta is None else None,
            target_colors=target_colors,
            K=K,
            use_dynamic_goals=use_dynamic,
            goal_positions=goal_positions,
            target_ids_by_step=target_ids_by_step,
            static_goals=static_goals if not use_dynamic else None,
        )
        print(f"Wrote {out_path} ({A} agents, {T} steps)")


if __name__ == "__main__":
    main()
