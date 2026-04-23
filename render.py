#!/usr/bin/env python
from __future__ import annotations

import sys
import os
import re

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import setproctitle
import numpy as np
from pathlib import Path

import torch

from config.config import (
    get_config,
    resolve_attn_comm_args,
    resolve_dynamic_target_reasoning_args,
    compute_attn_comm_robot_obs_dim,
    compute_attn_comm_tail_dim,
    compute_dynamic_robot_obs_dim,
    compute_dynamic_robot_obs_dim_legacy_full_k,
    compute_undetermined_robot_obs_dim,
    compute_undetermined_v2_robot_obs_dim,
    compute_undetermined_v2_attn_hybrid_robot_obs_dim,
    infer_dynamic_pack_from_actor_feat_dim,
)

from envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from runner.checkpoint_paths import (
    resolve_shared_actor_checkpoint_path,
    resolve_shared_critic_checkpoint_path,
)


def _parse_train_run_id_from_model_dir(model_dir: str) -> str | None:
    m = re.search(r"[/\\]train[/\\]run(\d+)", str(model_dir), flags=re.IGNORECASE)
    return m.group(1) if m else None


def _write_render_run_flags(
    repo_root: Path,
    run_dir: Path,
    curr_run: str,
    all_args,
    device: torch.device,
) -> None:
    """Mirror train's run_flags: record checkpoint + optional train run id; also under fig/render/<n>/."""
    md = str(getattr(all_args, "model_dir", "") or "")
    actor_path = resolve_shared_actor_checkpoint_path(md)
    ap_resolved = ""
    if actor_path is not None and actor_path.is_file():
        ap_resolved = str(actor_path.resolve())
    critic_path = resolve_shared_critic_checkpoint_path(md, actor_path)
    cp_resolved = ""
    if critic_path is not None and critic_path.is_file():
        cp_resolved = str(critic_path.resolve())
    train_rid = _parse_train_run_id_from_model_dir(md)
    lines = [
        f"render_results_subdir: {curr_run}\n",
        f"model_dir: {md}\n",
        f"actor_checkpoint: {ap_resolved}\n",
        f"critic_checkpoint: {cp_resolved}\n",
        f"train_run_id: {train_rid if train_rid is not None else '-'}\n",
        f"device: {device}\n",
        f"num_agents: {int(getattr(all_args, 'num_agents', 0))}\n",
        f"render_episodes: {int(getattr(all_args, 'render_episodes', 0))}\n",
        f"episode_length: {int(getattr(all_args, 'episode_length', 0))}\n",
        f"algorithm_name: {getattr(all_args, 'algorithm_name', '')}\n",
        f"no_render_auto_align_checkpoint: {bool(getattr(all_args, 'no_render_auto_align_checkpoint', False))}\n",
    ]
    text = "".join(lines)
    flags_path = run_dir / "run_flags.txt"
    flags_path.write_text(text, encoding="utf-8")
    try:
        n = int(str(curr_run).lower().replace("run", ""))
    except ValueError:
        n = None
    if n is not None:
        fig_dir = repo_root / "fig" / "render" / str(n)
        fig_dir.mkdir(parents=True, exist_ok=True)
        (fig_dir / "run_flags.txt").write_text(text, encoding="utf-8")


def apply_dynamic_goal_obs_dim(all_args):
    """Match DiscreteActionEnv / training: dynamic mode needs larger robot_obs_dim."""
    if getattr(all_args, "use_attn_comm_actor", False):
        return
    if getattr(all_args, "enable_dynamic_goal_assignment", False):
        k = int(all_args.num_agents)
        ver = getattr(all_args, "dynamic_obs_pack_version", "slots")
        if ver == "legacy":
            all_args.robot_obs_dim = compute_dynamic_robot_obs_dim_legacy_full_k(k)
        else:
            all_args.robot_obs_dim = compute_dynamic_robot_obs_dim(
                k,
                int(getattr(all_args, "dynamic_target_slot_count", 1)),
                use_neighbor_attn_lstm_actor=bool(
                    getattr(all_args, "use_neighbor_attn_lstm_actor", False)
                ),
                actor_neighbor_n=int(getattr(all_args, "actor_neighbor_n", 10)),
            )


def align_render_args_from_actor_checkpoint(all_args):
    """
    Read results/.../models/actor.pt and set enable_dynamic_goal_assignment,
    num_agents, and robot_obs_dim so the policy matches the checkpoint.
    """
    actor_path = resolve_shared_actor_checkpoint_path(getattr(all_args, "model_dir", None))
    if actor_path is None or not actor_path.is_file():
        print(f"[render] auto_align: no actor checkpoint under model_dir={getattr(all_args, 'model_dir', None)!r}, skip")
        return
    try:
        sd = torch.load(actor_path, map_location="cpu")
    except Exception as e:
        print(f"[render] auto_align: failed to load actor.pt: {e}")
        return
    feat_key = "base_robot.feature_norm.weight"
    if feat_key not in sd and "attn_comm_encoder.embed_self.weight" in sd:
        all_args.use_attn_comm_actor = True
        all_args.enable_dynamic_goal_assignment = False
        P = int(sd["attn_comm_encoder._attn_comm_p"].item())
        H = int(sd["attn_comm_encoder._attn_comm_h"].item())
        K = int(getattr(all_args, "num_agents", 10))
        Mkey = "attn_comm_encoder.embed_recv.weight"
        M = int(sd[Mkey].shape[1]) if Mkey in sd else int(getattr(all_args, "attn_comm_message_dim", 16))
        all_args.attn_comm_message_dim = M
        d_emb = int(sd["attn_comm_encoder.embed_self.weight"].shape[0])
        hs = int(getattr(all_args, "hidden_size", d_emb))
        if d_emb != hs:
            all_args.attn_comm_hidden_dim = d_emb
        all_args.attn_comm_ally_slots = P
        all_args.attn_comm_human_slots = H
        gh = "undetermined_head.goal_mlp.0.weight"
        if gh in sd and str(getattr(all_args, "architecture_mode", "")) == "attn_undetermined_goal":
            m = max(1, int(getattr(all_args, "undetermined_v2_goal_slots", 10)))
            all_args.enable_undetermined_goal = True
            all_args.enable_undetermined_goal_v2 = True
            rod = compute_undetermined_v2_attn_hybrid_robot_obs_dim(m, P, H, M)
            all_args.robot_obs_dim = rod
            tail = compute_attn_comm_tail_dim(P, H, M)
            print(
                f"[render] auto_align: attn_undetermined_goal hybrid M={m} P={P} H={H} "
                f"tail={tail} robot_obs_dim={rod} num_agents={K}"
            )
            return
        rod = compute_attn_comm_robot_obs_dim(P, H, message_dim=M)
        all_args.robot_obs_dim = rod
        print(
            f"[render] auto_align: attn_comm actor (fixed targets) P={P} H={H} "
            f"num_agents={K} robot_obs_dim={rod}"
        )
        return
    if feat_key not in sd:
        print("[render] auto_align: unexpected actor layout, skip")
        return
    D = int(sd[feat_key].shape[0])
    third_bias_key = "act.action_outs.2.linear.bias"
    if third_bias_key in sd:
        all_args.enable_dynamic_goal_assignment = True
        K = int(sd[third_bias_key].shape[0])
        all_args.num_agents = K
        pref_m = int(getattr(all_args, "dynamic_target_slot_count", K))
        pack, m_slot, p_nbr, rod = infer_dynamic_pack_from_actor_feat_dim(
            D, K, preferred_slot_m=pref_m
        )
        if pack is None:
            print(
                f"[render] auto_align WARN: feat dim {D} does not match known dynamic layouts "
                f"(K={K}); using current slot-based dim from args (load may fail)."
            )
            all_args.dynamic_obs_pack_version = "slots"
            all_args.robot_obs_dim = compute_dynamic_robot_obs_dim(
                K,
                int(getattr(all_args, "dynamic_target_slot_count", 1)),
                use_neighbor_attn_lstm_actor=bool(
                    getattr(all_args, "use_neighbor_attn_lstm_actor", False)
                ),
                actor_neighbor_n=int(getattr(all_args, "actor_neighbor_n", 10)),
            )
        elif pack == "legacy":
            all_args.dynamic_obs_pack_version = "legacy"
            all_args.use_neighbor_attn_lstm_actor = False
            all_args.robot_obs_dim = rod
            print(f"[render] auto_align: dynamic goals (legacy full-K obs), num_agents={K}")
        elif pack == "slots_attn":
            all_args.dynamic_obs_pack_version = "slots"
            all_args.use_neighbor_attn_lstm_actor = True
            all_args.dynamic_target_slot_count = int(m_slot)
            all_args.actor_neighbor_n = int(p_nbr)
            all_args.robot_obs_dim = rod
            print(
                f"[render] auto_align: dynamic goals (M={m_slot} slots + neighbor P={p_nbr}), "
                f"num_agents={K}, robot_obs_dim={rod}"
            )
        else:
            all_args.dynamic_obs_pack_version = "slots"
            all_args.use_neighbor_attn_lstm_actor = False
            all_args.dynamic_target_slot_count = int(m_slot)
            all_args.robot_obs_dim = rod
            print(
                f"[render] auto_align: dynamic goals (M={m_slot} slots), num_agents={K}, "
                f"robot_obs_dim={rod}"
            )
        if all_args.robot_obs_dim + 2 != D:
            print(
                f"[render] auto_align WARN: ckpt feat dim {D} vs "
                f"robot_obs_dim+2={all_args.robot_obs_dim + 2} (num_agents={K})"
            )
        print(
            f"[render] auto_align: actor input dim {D}, robot_obs_dim={all_args.robot_obs_dim}"
        )
    else:
        all_args.enable_dynamic_goal_assignment = False
        gh = "undetermined_head.goal_mlp.0.weight"
        pair_k = "undetermined_head.mlp.0.weight"
        if pair_k in sd and gh not in sd:
            all_args.enable_undetermined_goal = True
            all_args.enable_undetermined_goal_v2 = True
            all_args.undet_v2_head_arch = "pair_mlp"
            in_d = int(sd[pair_k].shape[1])
            m_slots = (in_d - 13) // 5
            if m_slots < 1 or (in_d - 13) % 5 != 0:
                m_slots = max(1, int(getattr(all_args, "undetermined_v2_goal_slots", 10)))
                print(
                    f"[render] auto_align WARN: pair_mlp in_d={in_d} does not match 13+5*M; "
                    f"using undetermined_v2_goal_slots={m_slots}"
                )
            all_args.undetermined_v2_goal_slots = m_slots
            all_args.robot_obs_dim = compute_undetermined_v2_robot_obs_dim(m_slots)
            if "undetermined_head.score_dir" in sd:
                all_args.undetermined_target_embed_dim = int(sd["undetermined_head.score_dir"].shape[0])
            print(
                f"[render] auto_align: undetermined goal v2 pair_mlp, M={m_slots}, "
                f"robot_obs_dim={all_args.robot_obs_dim}, undetermined_target_embed_dim="
                f"{int(getattr(all_args, 'undetermined_target_embed_dim', 96))} (actor input dim {D})"
            )
        elif gh in sd:
            all_args.enable_undetermined_goal = True
            in_f = int(sd[gh].shape[1])
            rod_bc = max(7, D - 2)
            if in_f == 5:
                all_args.enable_undetermined_goal_v2 = True
                m_slots = (rod_bc - 8) // 5
                if m_slots < 1 or (rod_bc - 8) % 5 != 0:
                    m_slots = max(1, int(getattr(all_args, "undetermined_v2_goal_slots", 10)))
                    print(
                        f"[render] auto_align WARN: undetermined v2 obs dim {rod_bc} does not match "
                        f"7+5*M+1; using undetermined_v2_goal_slots={m_slots}"
                    )
                all_args.undetermined_v2_goal_slots = m_slots
                all_args.robot_obs_dim = compute_undetermined_v2_robot_obs_dim(m_slots)
                print(
                    f"[render] auto_align: undetermined goal v2, M={m_slots}, "
                    f"robot_obs_dim={all_args.robot_obs_dim} (actor input dim {D})"
                )
            else:
                all_args.enable_undetermined_goal_v2 = False
                k_inf = (rod_bc - 7) // 5
                if k_inf >= 1 and rod_bc == 7 + 5 * k_inf:
                    all_args.num_agents = k_inf
                else:
                    k_inf = int(getattr(all_args, "num_agents", 10))
                    all_args.num_agents = k_inf
                    print(
                        f"[render] auto_align WARN: undetermined v1 K infer from rod={rod_bc} failed; "
                        f"using num_agents={k_inf}"
                    )
                all_args.robot_obs_dim = compute_undetermined_robot_obs_dim(int(all_args.num_agents))
                print(
                    f"[render] auto_align: undetermined goal v1, K={int(all_args.num_agents)}, "
                    f"robot_obs_dim={all_args.robot_obs_dim} (actor input dim {D})"
                )
        else:
            all_args.robot_obs_dim = max(7, D - 2)
            print(
                f"[render] auto_align: non-dynamic, robot_obs_dim={all_args.robot_obs_dim} "
                f"(actor input dim {D})"
            )


def _warn_checkpoint_dynamic_goal_head_mismatch(all_args):
    if not getattr(all_args, "enable_dynamic_goal_assignment", False):
        return
    actor_path = resolve_shared_actor_checkpoint_path(getattr(all_args, "model_dir", None))
    if actor_path is None or not actor_path.is_file():
        return
    try:
        sd = torch.load(actor_path, map_location="cpu")
    except Exception:
        return
    key = "act.action_outs.2.linear.bias"
    if key not in sd:
        return
    k_ckpt = int(sd[key].shape[0])
    n = int(all_args.num_agents)
    if k_ckpt != n:
        print(
            f"[render] WARN: checkpoint third-action head K={k_ckpt} != num_agents={n} "
            f"after nearest_n_radius pattern align; policy load may fail unless K matches."
        )


def align_num_agents_to_font_pattern_if_nearest_n_radius(all_args):
    """
    When critic uses nearest_n_radius slots, set num_agents to the selected render
    pattern's coordinate count (same selection as EnvCore), independent of dynamic_target.
    Runs after checkpoint auto-align so pattern length overrides ckpt K for env sizing.
    """
    if getattr(all_args, "agent_state_mode", "all") != "nearest_n_radius":
        return
    from envs.env_core import select_font_pattern_targets_for_args

    saved_rank = int(getattr(all_args, "env_rank", 0))
    all_args.env_rank = 0
    try:
        name, coords = select_font_pattern_targets_for_args(all_args, log_selection=False)
    finally:
        all_args.env_rank = saved_rank

    n = len(coords)
    prev = int(all_args.num_agents)
    if n != prev:
        print(
            f"[render] nearest_n_radius: num_agents {prev} -> {n} "
            f"(pattern '{name}', {n} coordinates)"
        )
    else:
        print(
            f"[render] nearest_n_radius: num_agents={n} (pattern '{name}')"
        )
    all_args.num_agents = n
    apply_dynamic_goal_obs_dim(all_args)
    _warn_checkpoint_dynamic_goal_head_mismatch(all_args)


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            from envs.env_discrete import DiscreteActionEnv
            all_args.env_rank = rank
            env = DiscreteActionEnv(all_args)

            env.seed(all_args.seed + rank * 1000 )
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)], all_args)
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def parser_args(args, parser):
    parser.add_argument('--num_agents', type=int,default=10, help="number of players")
    parser.add_argument("--random_act_prob", type=int, default=0, help="the probability of robot to choice random action")
    parser.add_argument(
        "--no_render_auto_align_checkpoint",
        action="store_true",
        default=False,
        help="Disable reading actor.pt to match dynamic goals / num_agents / robot_obs_dim (auto-align is on by default).",
    )

    all_args = parser.parse_known_args(args)[0]
    resolve_dynamic_target_reasoning_args(all_args)
    if getattr(all_args, "use_attn_comm_actor", False):
        resolve_attn_comm_args(all_args)
        all_args.robot_obs_dim = compute_attn_comm_robot_obs_dim(
            int(all_args.attn_comm_ally_slots),
            int(all_args.attn_comm_human_slots),
            message_dim=int(getattr(all_args, "attn_comm_message_dim", 16)),
        )
    if getattr(all_args, "enable_dynamic_goal_assignment", False):
        all_args.use_neighbor_attn_lstm_actor = not getattr(
            all_args, "disable_neighbor_attn_lstm_actor", False
        )
    else:
        all_args.use_neighbor_attn_lstm_actor = False

    apply_dynamic_goal_obs_dim(all_args)

    return all_args


def main(args):
    parser = get_config()
    all_args = parser_args(args, parser)
    all_args.use_render = True
    # Shared-policy checkpoint: folder with actor.pt, or 4.pt if actor.pt is missing, or a direct path to *.pt.
    _default_model_dir = str(Path(__file__).resolve().parent / "results/train/run6/models")
    if not getattr(all_args, "model_dir", None):
        all_args.model_dir = _default_model_dir
    all_args.n_rollout_threads = 1
    all_args.episode_length = 500
    all_args.visualize = False
    all_args.render_episodes = 100
    all_args.num_attention_agents = 10
    all_args.num_humans = 2
    all_args.method = 'ppo'

    if all_args.method == 'ppo':
        if all_args.algorithm_name == "rmappo":
            print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
            all_args.use_recurrent_policy = True
            all_args.use_naive_recurrent_policy = False
        elif all_args.algorithm_name == "mappo":
            print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
            all_args.use_recurrent_policy = False 
            all_args.use_naive_recurrent_policy = False
        elif all_args.algorithm_name == "ippo":
            print("u are choosing to use ippo, we set use_centralized_V to be False.")
            all_args.use_centralized_V = False
        else:
            raise NotImplementedError
    elif all_args.method == 'orca':
        print('u chose orca method!')
    elif all_args.method == 'apf':
        print('u chose ipf method!')

    assert all_args.use_render, ("u need to set use_render be True")
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    assert all_args.n_rollout_threads==1, ("only support to use 1 env to render.")

    # After model_dir is set: match checkpoint (dynamic target head + obs dim) unless disabled.
    if not getattr(all_args, "no_render_auto_align_checkpoint", False):
        align_render_args_from_actor_checkpoint(all_args)
    if getattr(all_args, "use_attn_comm_actor", False):
        resolve_attn_comm_args(all_args)
    apply_dynamic_goal_obs_dim(all_args)
    align_num_agents_to_font_pattern_if_nearest_n_radius(all_args)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results" + "/render")
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    print(f'================ Results Saved in {curr_run} ================\n')
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    _write_render_run_flags(Path(_REPO_ROOT).resolve(), run_dir, curr_run, all_args, device)

    setproctitle.setproctitle("@" + str(all_args.user_name))     #进程名称

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    if True:
        from runner.shared.env_runner import EnvRunner as Runner

        runner = Runner(config)
        runner.render()
    else:
        from runner.separated.env_runner import EnvRunner as Runner

        runner = Runner(config)
        runner.render(mode='vedio', visualize=all_args.visualize, method=all_args.method)



    
    
    # post process
    # envs.close()
    print('accomplish test,test episode:',all_args.render_episodes)

if __name__ == "__main__":
    main(sys.argv[1:])
