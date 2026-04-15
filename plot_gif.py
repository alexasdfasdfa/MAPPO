#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制智能体运动的GIF动画
显示：初始位置、目标位置、移动路径
运动结束后停留5秒再循环

使用方法:
    python plot_gif.py --render_run 4 --episode 1
    python plot_gif.py --render_run 4 --episode 1 --output my_animation.gif # 输出指定文件名
    python plot_gif.py --render_run 4 --episode 1 --pause 5 # 停留时间（秒）
"""

import os
import sys
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]


def parse_coords_file(filepath):
    episodes = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'(\d+),\s*(.+)', line)
            if match:
                ep_num = int(match.group(1))
                coords_str = match.group(2)
                coords = []
                points = coords_str.split(';')
                for point in points:
                    point = point.strip()
                    if point:
                        parts = point.split()
                        if len(parts) >= 2:
                            x, y = float(parts[0]), float(parts[1])
                            coords.append((x, y))
                episodes[ep_num] = coords
    return episodes


def load_all_agents_coords(coords_dir, episode_num):
    all_trajectories = []
    agent_files = sorted([f for f in os.listdir(coords_dir) if f.startswith('coords_agent')])
    for agent_file in agent_files:
        filepath = os.path.join(coords_dir, agent_file)
        episodes = parse_coords_file(filepath)
        if episode_num in episodes:
            all_trajectories.append(episodes[episode_num])
    return all_trajectories


def get_formation_targets(formation_name, num_agents, scale=5.0):
    targets = []
    if formation_name == 'line':
        for i in range(num_agents):
            x = (i - num_agents/2 + 0.5) * scale / num_agents * 3
            y = scale
            targets.append((x, y))
    elif formation_name == 'circle':
        for i in range(num_agents):
            angle = 2 * np.pi * i / num_agents
            x = scale * np.cos(angle)
            y = scale * np.sin(angle)
            targets.append((x, y))
    elif formation_name == 'triangle':
        import math
        targets.append((0, scale * 1.5))
        if num_agents > 1:
            targets.append((-scale/2, scale * 0.75))
            targets.append((scale/2, scale * 0.75))
        for i in range(num_agents - 3):
            x = -scale + (i * 2 * scale / max(1, num_agents - 4))
            y = 0
            targets.append((x, y))
    elif formation_name == 'square':
        side = int(np.ceil(np.sqrt(num_agents)))
        for i in range(num_agents):
            row = i // side
            col = i % side
            x = (col - side/2 + 0.5) * scale / side
            y = (row - side/2 + 0.5) * scale / side + scale/2
            targets.append((x, y))
    else:
        return None
    return targets[:num_agents]


def create_animation(trajectories, targets=None, initial_pos=None,
                     interval=50, trail_length=10, pause_duration=5.0):
    if not trajectories:
        raise ValueError("No trajectory data")

    num_agents = len(trajectories)
    max_steps = max(len(t) for t in trajectories)

    # Calculate pause frames (e.g., 5 seconds * 1000ms / interval)
    pause_frames = int(pause_duration * 1000 / interval)
    total_frames = max_steps + pause_frames

    if targets is None:
        targets = [t[-1] if t else (0, 0) for t in trajectories]

    if initial_pos is None:
        initial_pos = [t[0] if t else (0, 0) for t in trajectories]

    fig, ax = plt.subplots(figsize=(12, 10))

    all_x = [p[0] for traj in trajectories for p in traj]
    all_y = [p[1] for traj in trajectories for p in traj]
    all_x.extend([t[0] for t in targets])
    all_y.extend([t[1] for t in targets])

    margin = 2
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Multi-Agent Formation Animation', fontsize=14)

    agent_circles = []
    trail_lines = []
    path_lines = []

    for i in range(num_agents):
        color = COLORS[i % len(COLORS)]
        circle = Circle((0, 0), 0.3, color=color, zorder=5)
        ax.add_patch(circle)
        agent_circles.append(circle)

        trail, = ax.plot([], [], color=color, linewidth=2, alpha=0.7, zorder=3)
        trail_lines.append(trail)

        full_path, = ax.plot([], [], color=color, linewidth=1, alpha=0.3, zorder=2)
        path_lines.append(full_path)

    target_markers = []
    for i, (tx, ty) in enumerate(targets):
        color = COLORS[i % len(COLORS)]
        marker, = ax.plot(tx, ty, marker='*', markersize=15, color=color,
                         markeredgecolor='black', markeredgewidth=0.5, zorder=4)
        target_markers.append(marker)

    init_markers = []
    for i, (ix, iy) in enumerate(initial_pos):
        color = COLORS[i % len(COLORS)]
        marker, = ax.plot(ix, iy, marker='s', markersize=10, color=color,
                         markeredgecolor='black', markeredgewidth=0.5, zorder=4,
                         fillstyle='none', linewidth=1.5)
        init_markers.append(marker)

    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    done_text = ax.text(0.5, 0.5, '', transform=ax.transAxes,
                        fontsize=16, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    legend_elements = [
        Line2D([0], [0], marker='s', color='gray', linestyle='None',
               markersize=10, markerfacecolor='none', label='Initial Position'),
        Line2D([0], [0], marker='*', color='gray', linestyle='None',
               markersize=15, label='Target Position'),
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=10, label='Agent')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    def init():
        for circle in agent_circles:
            circle.center = (0, 0)
        for trail in trail_lines:
            trail.set_data([], [])
        for path in path_lines:
            path.set_data([], [])
        time_text.set_text('')
        done_text.set_text('')
        return agent_circles + trail_lines + path_lines + [time_text, done_text]

    def animate(frame):
        actual_frame = min(frame, max_steps - 1)
        is_paused = frame >= max_steps

        for i, traj in enumerate(trajectories):
            if actual_frame < len(traj):
                x, y = traj[actual_frame]
            else:
                x, y = traj[-1] if traj else (0, 0)

            agent_circles[i].center = (x, y)

            if is_paused:
                full_x = [p[0] for p in traj]
                full_y = [p[1] for p in traj]
                path_lines[i].set_data(full_x, full_y)
                trail_lines[i].set_data(full_x, full_y)
            else:
                start_idx = max(0, actual_frame - trail_length)
                if i < len(trajectories):
                    trail_x = [trajectories[i][j][0] for j in range(start_idx, min(actual_frame+1, len(trajectories[i])))]
                    trail_y = [trajectories[i][j][1] for j in range(start_idx, min(actual_frame+1, len(trajectories[i])))]
                    trail_lines[i].set_data(trail_x, trail_y)

                    full_x = [trajectories[i][j][0] for j in range(min(actual_frame+1, len(trajectories[i])))]
                    full_y = [trajectories[i][j][1] for j in range(min(actual_frame+1, len(trajectories[i])))]
                    path_lines[i].set_data(full_x, full_y)

        if is_paused:
            remaining_pause = (pause_frames - (frame - max_steps)) * interval / 1000
            time_text.set_text('Step: {}/{} (Completed)'.format(max_steps-1, max_steps-1))
            # done_text.set_text('Formation Complete!\nResetting in {:.1f}s...'.format(remaining_pause))
        else:
            time_text.set_text('Step: {}/{}'.format(actual_frame, max_steps-1))
            done_text.set_text('')

        return agent_circles + trail_lines + path_lines + [time_text, done_text]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=total_frames, interval=interval,
                                   blit=True, repeat=True)

    return fig, anim


def main():
    parser = argparse.ArgumentParser(description='Generate agent motion GIF animation')
    parser.add_argument('--render_run', type=int, default=4,
                        help='Render run number (default: 4)')
    parser.add_argument('--episode', type=int, default=1,
                        help='Episode number to visualize (default: 1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: agent_motion_run{run}_ep{ep}.gif)')
    parser.add_argument('--interval', type=int, default=50,
                        help='Frame interval in ms (default: 50)')
    parser.add_argument('--trail_length', type=int, default=20,
                        help='Trail length (default: 20)')
    parser.add_argument('--formation', type=str, default=None,
                        help='Formation shape (line/circle/triangle/square)')
    parser.add_argument('--pause', type=float, default=5.0,
                        help='Pause duration after motion ends in seconds (default: 5.0)')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    coords_dir = os.path.join(base_dir, 'results/render/run{}/coords'.format(args.render_run))

    if not os.path.exists(coords_dir):
        print("Error: Coords directory not found: {}".format(coords_dir))
        sys.exit(1)

    print("Loading coords data: {}".format(coords_dir))
    trajectories = load_all_agents_coords(coords_dir, args.episode)

    if not trajectories:
        print("Error: No data found for episode {}".format(args.episode))
        sys.exit(1)

    print("Found {} agents' trajectories".format(len(trajectories)))
    print("Trajectory lengths: {}...".format([len(t) for t in trajectories[:5]]))

    if args.formation:
        targets = get_formation_targets(args.formation, len(trajectories))
        print("Using formation shape: {}".format(args.formation))
    else:
        targets = None
        print("Using trajectory endpoints as target positions")

    initial_pos = [t[0] if t else (0, 0) for t in trajectories]

    print("Creating animation (pause {}s after completion)...".format(args.pause))
    fig, anim = create_animation(
        trajectories,
        targets=targets,
        initial_pos=initial_pos,
        interval=args.interval,
        trail_length=args.trail_length,
        pause_duration=args.pause
    )

    if args.output:
        output_file = args.output
    else:
        output_dir = os.path.join(base_dir, 'fig', 'motion')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir,
                                   'agent_motion_run{}_ep{}.gif'.format(args.render_run, args.episode))

    print("Saving animation to: {}".format(output_file))
    anim.save(output_file, writer='pillow', fps=20)
    print("Done!")

    plt.close(fig)


if __name__ == '__main__':
    main()