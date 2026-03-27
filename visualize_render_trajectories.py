#!/usr/bin/env python3
"""
Build one multi-agent trajectory GIF per episode from render coord logs.

Reads:  <repo>/results/render/run{n}/coords/coords_agent*.txt,
        run{n}/episode_meta.jsonl (pattern, lengths, per-agent goals; optional for old runs)
Writes: <cwd>/fig/render/n/{ep_id}.gif  (all agents + hollow goal markers when meta exists)

Example:
  python visualize_render_trajectories.py 13
  python visualize_render_trajectories.py 13 --episode 3 --stride 4   # fewer frames, faster
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


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
    files = sorted(coords_dir.glob("coords_agent*.txt"))
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
    colors: np.ndarray,
    episode_meta: dict | None,
    header_fallback: dict | None,
) -> None:
    """traj: (T, A, 2); colors: (A, 4) RGBA from colormap."""
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

    goals = None
    if episode_meta and isinstance(episode_meta.get("agent_goals"), list):
        goals = episode_meta["agent_goals"]
    if goals:
        for i, g in enumerate(goals):
            if len(g) < 2:
                continue
            hex_c = matplotlib.colors.to_hex(colors[i % A])
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
        hex_c = matplotlib.colors.to_hex(colors[i])
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

    for ep_id, traj in to_run:
        T, A, _ = traj.shape
        ep_meta = meta_by_ep.get(ep_id)
        extra = None
        if ep_meta and isinstance(ep_meta.get("agent_goals"), list):
            extra = np.array(ep_meta["agent_goals"], dtype=np.float64)
        bounds = _axis_bounds(traj, extra_xy=extra)
        colors = plt.cm.tab10(np.linspace(0, 1, max(A, 10)))[:A]
        out_path = out_dir / f"{ep_id}.gif"
        render_all_agents_gif(
            traj,
            bounds,
            out_path,
            episode_id=ep_id,
            stride=args.stride,
            dpi=args.dpi,
            duration=args.duration,
            colors=colors,
            episode_meta=ep_meta,
            header_fallback=header_fallback if ep_meta is None else None,
        )
        print(f"Wrote {out_path} ({A} agents, {T} steps)")


if __name__ == "__main__":
    main()
