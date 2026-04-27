import argparse
from typing import Optional


def get_config():
    parser = argparse.ArgumentParser(
        description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="mappo", choices=["rmappo", "mappo"])

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    parser.add_argument("--n_training_threads",type=int,default=2,help="Number of torch threads for training",)
    parser.add_argument("--n_rollout_threads",type=int,default=100,help="Number of parallel envs for training rollouts",)
    parser.add_argument("--n_eval_rollout_threads",type=int,default=2,help="Number of parallel envs for evaluating rollouts",)
    parser.add_argument("--n_render_rollout_threads",type=int,default=1,help="Number of parallel envs for rendering rollouts",)
    parser.add_argument("--num_env_steps",type=int,default=10000000 ,help="Number of environment steps to train (default: 10e6)",)
    parser.add_argument("--user_name",type=str,default="marl",help="[for wandb usage], to specify user's name for simply collecting training data.",)

    # env parameters
    parser.add_argument("--env_name", type=str, default="MyEnv", help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state",action="store_true",default=False,help="Whether to use global state or concatenated obs")
    parser.add_argument("--time_step",type=float,default=0.1,help='The time interval between each step')
    parser.add_argument("--robot_obs_dim",type=int,default=7,help="robot observation dimension")
    parser.add_argument("--human_obs_dim",type=int,default=5,help="human observation dimension")
    parser.add_argument("--vel_action_dim",type=int,default=5,help="dimension of velocity action space")
    parser.add_argument("--dir_action_dim",type=int,default=18,help="dimension of direction action space")

    # Dynamic multi-goal assignment (optional)
    parser.add_argument(
        "--enable_dynamic_goal_assignment",
        action="store_true",
        default=False,
        help="Robots choose a discrete target index each step; claims + path-length / conflict shaping rewards.",
    )
    parser.add_argument(
        "--enable_undetermined_goal",
        action="store_true",
        default=True,
        help="Undetermined goal mode: sticky targets, embedding-based selection + auction, Hungarian shaping once per "
        "assignment; motion is dir+vel only (like non-dynamic). Mutually exclusive with --enable_dynamic_goal_assignment.",
    )
    parser.add_argument(
        "--undetermined_obs_goal_radius",
        type=float,
        default=5.0,
        help="Within this distance to a goal center, agent observes claimed status for that goal.",
    )
    parser.add_argument(
        "--undetermined_comm_radius",
        type=float,
        default=6.0,
        help="Local communication radius: agents this close with same target trigger auction tie-break.",
    )
    parser.add_argument(
        "--undetermined_hungarian_reward_scale",
        type=float,
        default=0.10,
        help="Per-agent reward scale: -scale * (current_assignment_cost - optimal_cost) / n once per assignment batch. "
        "Train script caps at 0.10 in undetermined mode so it does not overwhelm navigation.",
    )
    parser.add_argument(
        "--undetermined_target_embed_dim",
        type=int,
        default=32,
        help="Embedding dim for undetermined target-selection head (dot-product preferences).",
    )
    parser.add_argument(
        "--undetermined_max_auction_rounds",
        type=int,
        default=8,
        help="Max rounds of target-selection + apply_undetermined_targets while any agent is pending.",
    )
    parser.add_argument(
        "--enable_undetermined_goal_v2",
        action="store_true",
        default=True,
        help="Undetermined v2: observation uses fixed M nearest-goal slots (not 4*K + (K-1) peer tids tied to swarm size); "
        "target head scores M slots then maps to global goal id. Requires --enable_undetermined_goal. "
        "Uses apply_undetermined_v2_reward_floors (softer avoid / distance shaping, approach bonus, Hungarian divisor decoupled from N).",
    )
    parser.add_argument(
        "--undetermined_v2_goal_slots",
        type=int,
        default=10,
        help="Number of nearest-goal slots packed into each robot obs in undetermined v2 (capped by K at runtime).",
    )
    parser.add_argument(
        "--enable_undetermined_v2_exchange",
        action="store_true",
        default=True,
        help="Undetermined v2 only: each env step, a heuristic may swap two agents' discrete targets when they lie "
        "within a local domain radius and the swap passes --undetermined_v2_exchange_accept_criterion "
        "(default fleet_m: fleet-wide M; pair_max: max(d_i,gi,d_j,gj)-max(d_i,gj,d_j,gi)>min_gain). "
        "Incompatible with --enable_undetermined_goal_v3.",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_radius",
        type=float,
        default=None,
        help="Pairwise exchange domain (meters): only agent pairs with center distance <= this are candidates. "
        "None defaults to --undetermined_comm_radius.",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_min_gain",
        type=float,
        default=0.05,
        help="Minimum improvement (meters) to accept a swap: under fleet_m, M_before-M_after for fleet M; "
        "under pair_max, max(d_i,gi,d_j,gj)-max(d_i,gj,d_j,gi) for the two agents.",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_max_pairs_per_step",
        type=int,
        default=1,
        help="fleet_m/pair_max: max disjoint swaps per step. cone_mutual_greedy_m: max greedy swap rounds "
        "(each round picks one pair that strictly lowers fleet M among unused agents).",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_ignore_pending",
        action="store_true",
        default=False,
        help="If set, allow swaps even when an agent has undetermined_target_pending; default skips such agents.",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_accept_criterion",
        type=str,
        default="fleet_m",
        choices=("fleet_m", "pair_max", "cone_mutual_greedy_m"),
        help="fleet_m: fleet M drop > min_gain. pair_max: pair max-dist gain > min_gain. "
        "cone_mutual_greedy_m: velocity-cone + strict mutual d(i,Tj)<d(i,Ti) & d(j,Ti)<d(j,Tj); then greedy "
        "disjoint swaps each minimizing fleet M (see cone_* and max_pairs_per_step).",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_cone_half_deg",
        type=float,
        default=60.0,
        help="cone_mutual_greedy_m: half-angle (degrees) of each agent's forward cone around its velocity; "
        "neighbor must lie strictly inside (dot > cos(half)).",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_forward_speed_eps",
        type=float,
        default=1e-3,
        help="cone_mutual_greedy_m: if speed < this, forward axis uses (cos(theta), sin(theta)).",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_bottleneck_shaping_scale",
        type=float,
        default=0.0,
        help="Undetermined v2 + v2_exchange only: per-step team shaping when fleet bottleneck distance "
        "M=max_k dist2goal shrinks (proxy for shortening the slowest agent's remaining path). "
        "Each agent gets (scale * ΔM) / (divisor * N) added to shaped reward; 0 disables.",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_team_dist_shaping_scale",
        type=float,
        default=0.0,
        help="Undetermined v2 + v2_exchange only: per-step team shaping when sum_k dist2goal shrinks "
        "(proxy for less total work to goals). Each agent gets (scale * ΔS) / (divisor * N); 0 disables.",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_swap_bonus_scale",
        type=float,
        default=0.0,
        help="Undetermined v2 + v2_exchange only: extra bonus for agents that participated in a target swap "
        "this step, from pre-swap to post-swap fleet M and total distance (see swap_sum_weight). 0 disables.",
    )
    parser.add_argument(
        "--undetermined_v2_exchange_shaping_team_divisor",
        type=float,
        default=8.0,
        help="Same role as undetermined_v2_hungarian_team_divisor: scales exchange shaping / swap bonus.",
    )
    # Exchange learning (behavior cloning to replace rule-based exchange)
    parser.add_argument(
        "--enable_exchange_learning",
        action="store_true",
        default=False,
        help="Enable exchange learning: collect data from rule-based exchange, train a neural network to mimic it.",
    )
    parser.add_argument(
        "--exchange_data_dir",
        type=str,
        default="./exchange_data",
        help="Directory for exchange training data and model checkpoints.",
    )
    parser.add_argument(
        "--exchange_train_interval",
        type=int,
        default=20000,
        help="Train exchange network every N total env steps (default 20000: ~400 rounds over 8M steps).",
    )
    parser.add_argument(
        "--exchange_accuracy_threshold",
        type=float,
        default=0.90,
        help="Validation accuracy threshold to switch from rule-based to neural exchange.",
    )
    parser.add_argument(
        "--exchange_lr",
        type=float,
        default=1e-3,
        help="Learning rate for exchange network training.",
    )
    parser.add_argument(
        "--exchange_train_epochs",
        type=int,
        default=50,
        help="Number of training epochs per exchange network training round.",
    )
    parser.add_argument(
        "--enable_exchange_network",
        action="store_true",
        default=False,
        help="Use trained exchange network instead of rule-based exchange. Requires exchange model in exchange_data_dir.",
    )
    parser.add_argument(
        "--exchange_swap_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for neural exchange: swap if predicted prob >= threshold.",
    )
    parser.add_argument(
        "--enable_undetermined_goal_v3",
        action="store_true",
        default=False,
        help="Undetermined v3: same nearest-M goal core and v2-style target head + rewards as v2, but observation always "
        "includes AttnComm tail (ally recv + messages + humans). Ally/human slot counts come from "
        "--undetermined_v3_comm_* (fixed caps), not from num_agents, so checkpoints/layout stay agent-count agnostic. "
        "Requires --enable_undetermined_goal, --use_attn_comm_actor, --architecture_mode attn_undetermined_goal, "
        "and disables --enable_undetermined_goal_v2.",
    )
    parser.add_argument(
        "--undetermined_v3_comm_ally_slots",
        type=int,
        default=6,
        help="Undetermined v3: max in-radius ally slots in the comm tail (decoupled from num_agents). "
        "Train syncs attn_comm_ally_slots to this when v3 is enabled.",
    )
    parser.add_argument(
        "--undetermined_v3_comm_human_slots",
        type=int,
        default=4,
        help="Undetermined v3: max in-radius human slots in the comm tail (decoupled from num_agents).",
    )
    parser.add_argument(
        "--undetermined_v3_attn_embed_dim",
        type=int,
        default=64,
        help="Undetermined v3: embedding width for scaled dot-product attention over M goal slots (target head).",
    )
    parser.add_argument(
        "--undetermined_v3_fuse_hidden",
        type=int,
        default=128,
        help="Undetermined v3: hidden units for fusion MLP that adds a residual to slot logits from "
        "(ego, prev target norm, slot features).",
    )
    parser.add_argument(
        "--undetermined_v3_target_kl_coef",
        type=float,
        default=0.0,
        help="Undetermined v3: KL( softmax(new_logits) || softmax(old_logits.detach)) on the M-slot target head, "
        "where old logits are those stored at rollout collection (trust-region / slow drift of target choice). "
        "0 disables extra buffer + loss.",
    )
    parser.add_argument(
        "--undet_v2_target_latent_model_dir",
        type=str,
        default=None,
        help="Sibling folder (resolved from MAPPO repo root): MAPPO actor.pt with undetermined_head.*, or "
        "target_latent_selector.pt when --undet_v2_head_arch pair_mlp (loads into undetermined_head). "
        "Empty string disables.",
    )
    parser.add_argument(
        "--undet_v2_latent_train_mode",
        type=str,
        default="motion_only",
        choices=["motion_only", "finetune_all"],
        help="With undet_v2_target_latent_model_dir: motion_only freezes undetermined_head (train navigation / "
        "base actor only). finetune_all keeps the head trainable and adds a small supervised slot loss during PPO "
        "(rollout target picks are no_grad, so the head needs this aux term to receive gradients).",
    )
    parser.add_argument(
        "--undet_v2_target_head_aux_coef",
        type=float,
        default=0.15,
        help="When undet_v2_latent_train_mode=finetune_all (undetermined v2, no attn actor): weight on auxiliary "
        "cross-entropy that matches the nearest-M slot to the goal implied by obs (gx,gy). Higher helps scratch "
        "training without latent pretrain. Set 0 to disable aux (head gets no PPO grads). Ignored in motion_only.",
    )
    parser.add_argument(
        "--undet_v2_head_arch",
        type=str,
        default="dot_product",
        choices=["dot_product", "pair_mlp"],
        help="Undetermined v2 target head: dot_product (ego_mlp+goal_mlp) or pair_mlp (same architecture as "
        "undet_v2_target_latent TargetLatentSelector; required to load target_latent_selector.pt). "
        "pair_mlp needs pure v2 obs (no AttnComm hybrid tail). For pair_mlp set --undetermined_target_embed_dim "
        "to match standalone d_emb (often 96).",
    )
    parser.add_argument(
        "--undet_v2_pair_mlp_hidden",
        type=int,
        default=384,
        help="Hidden width in pair_mlp head (match undet_v2_target_latent --hidden).",
    )
    parser.add_argument(
        "--undet_v2_pair_mlp_no_layernorm",
        action="store_true",
        default=False,
        help="Disable LayerNorm in pair_mlp head (match standalone --no-layernorm).",
    )
    parser.add_argument(
        "--undet_v2_pair_coord_scale",
        type=float,
        default=10.0,
        help="coord_scale for rel/cs in pair_mlp forward (match undet_v2_target_latent --box).",
    )
    parser.add_argument(
        "--undet_v2_pair_dist_box",
        type=float,
        default=10.0,
        help="Box length for learned dist bias in pair_mlp (match training box side).",
    )
    parser.add_argument(
        "--undetermined_v2_hungarian_team_divisor",
        type=float,
        default=8.0,
        help="Undetermined v2: Hungarian per-agent bonus uses -scale*gap/divisor instead of dividing by num_agents.",
    )
    parser.add_argument(
        "--undetermined_v2_avoid_exp_cap",
        type=float,
        default=2.0,
        help="Undetermined v2: cap |r_avoid_raw| from exp discomfort so nd_discount_avoid does not dominate returns.",
    )
    parser.add_argument(
        "--undetermined_v2_discount_avoid_mult",
        type=float,
        default=0.42,
        help="Undetermined v2: multiply nd_discount_avoid by this factor (after floors).",
    )
    parser.add_argument(
        "--undetermined_v2_dist_penalty_mult",
        type=float,
        default=0.38,
        help="Undetermined v2: scale undetermined linear+quadratic distance penalties on r_nav.",
    )
    parser.add_argument(
        "--undetermined_v2_approach_reward_scale",
        type=float,
        default=0.85,
        help="Undetermined v2: dense bonus scale * clip(pre_d - d, 0, cap) toward assigned goal (in addition to nd progress).",
    )
    parser.add_argument(
        "--undetermined_v2_approach_reward_cap",
        type=float,
        default=0.55,
        help="Max per-step approach shaping (world units) after clip, before nd_discount_nav.",
    )
    parser.add_argument(
        "--undetermined_v2_sl_dense_scale",
        type=float,
        default=0.28,
        help="Undetermined v2 only: dense type-2 (formation) reward = scale * clip(S_L,0,1) added to r_nav, "
        "S_L=1-||L_hat-L_des||_F/||L_des||_F (L_des from goal sites — topological hint vs literal waypoint chasing). 0=off.",
    )
    parser.add_argument(
        "--undetermined_v2_sl_success_scale",
        type=float,
        default=2.5,
        help="Undetermined v2 only: extra bonus on r_nav when S_L >= --undetermined_v2_sl_success_threshold "
        "(type-2 / similarity success). 0=off.",
    )
    parser.add_argument(
        "--undetermined_v2_sl_success_threshold",
        type=float,
        default=0.97,
        help="S_L threshold for --undetermined_v2_sl_success_scale (similarity / formation success).",
    )
    parser.add_argument(
        "--undetermined_v2_sl_delta_scale",
        type=float,
        default=0.0,
        help="Undetermined v2 type-2: add scale * max(0, S_L - S_L_prev) to dense shaping (same channel as sl_raw). "
        "Rewards improving Laplacian match vs previous step; goals induce L_des as a topological hint.",
    )
    parser.add_argument(
        "--undetermined_v2_type2_pattern_first",
        action="store_true",
        default=False,
        help="Undetermined v2: after floors, weaken literal (gx,gy) anchoring (approach + dist penalties + progress) "
        "and strengthen type-2 (raise S_L dense/success caps, set sl_delta_scale if still 0). Use for class-2 / pattern emphasis.",
    )
    parser.add_argument(
        "--undetermined_v2_type2_formation_efficiency",
        action="store_true",
        default=True,
        help="Undetermined v2: preset for type-2 success — relax literal goal pull and S_L shaping after S_L>=thr "
        "(allow formation drift vs targets), reward first crossing into success, add light pre-success step/travel cost "
        "to favor fewer steps and shorter motion before formation.",
    )
    parser.add_argument(
        "--undetermined_v2_sl_post_success_literal_scale",
        type=float,
        default=1.0,
        help="Undetermined v2: when laplacian S_L >= success threshold, scale (gx,gy) shaping: progress, approach, "
        "dist penalties, proximity, heading, and nd goal-disk raw. 1.0=no change; ~0.25–0.35 allows drift after formation.",
    )
    parser.add_argument(
        "--undetermined_v2_sl_post_success_sl_shaping_scale",
        type=float,
        default=1.0,
        help="Undetermined v2: when S_L >= success threshold, scale S_L dense + delta terms (so small post-success "
        "S_L dips from drift are less punitive). 1.0=no change.",
    )
    parser.add_argument(
        "--undetermined_v2_sl_success_only_on_crossing",
        action="store_true",
        default=False,
        help="Undetermined v2: pay undetermined_v2_sl_success_scale mainly when S_L first crosses the threshold "
        "(S_L_prev < thr <= S_L); optional sustain via --undetermined_v2_sl_success_sustain_frac.",
    )
    parser.add_argument(
        "--undetermined_v2_sl_success_sustain_frac",
        type=float,
        default=0.0,
        help="Undetermined v2: when --undetermined_v2_sl_success_only_on_crossing, add success_scale*sustain_frac "
        "on subsequent steps while S_L>=thr (0 = no per-step sustain).",
    )
    parser.add_argument(
        "--undetermined_v2_sl_pre_success_step_penalty",
        type=float,
        default=0.0,
        help="Undetermined v2: add this raw value to r_nav each step while S_L < success threshold (use small negative "
        "e.g. -0.02 to encourage fewer iterations before formation). 0=off.",
    )
    parser.add_argument(
        "--undetermined_v2_sl_pre_success_travel_penalty",
        type=float,
        default=0.0,
        help="Undetermined v2: each step while S_L < thr, add this * (team step travel sum / N) to r_nav (negative "
        "values penalize motion before formation). Requires undetermined/dynamic path sync travel. 0=off.",
    )
    parser.add_argument(
        "--undetermined_far_goal_progress_dist_thresh",
        type=float,
        default=4.0,
        help="If dist2goal > this, multiply nd_goal_progress_coef by undetermined_far_goal_progress_boost (undetermined only).",
    )
    parser.add_argument(
        "--undetermined_far_goal_progress_boost",
        type=float,
        default=1.12,
        help="Multiplier on progress coef when far from assigned (gx,gy) (undetermined only). Keep near 1 to limit far-field nav exploit.",
    )
    parser.add_argument(
        "--undetermined_goal_distance_penalty_scale",
        type=float,
        default=0.012,
        help="Subtract scale*dist2goal from r_nav raw each step (undetermined); anchors world position vs relative formation. 0=off.",
    )
    parser.add_argument(
        "--undetermined_goal_dist_penalty_quad_scale",
        type=float,
        default=0.00012,
        help="Undetermined: additionally subtract scale*dist2goal^2 from r_nav each step (0=off).",
    )
    parser.add_argument(
        "--architecture_mode",
        type=str,
        default="default",
        choices=["default", "attn_undetermined_goal"],
        help="attn_undetermined_goal: Cons-DecAF-style stack — undetermined goal v2/v3 nearest-M core unchanged; "
        "ConsMAC via AttnComm tail; CTDE MAPPO + nearest_n_radius critic; reward Eq.(1)-(4) in Xiang et al. "
        "(arXiv:2307.12287). Use with --enable_undetermined_goal_v3 for v3 (comm tail P/H decoupled from num_agents).",
    )
    parser.add_argument(
        "--cons_mac_ce_bins",
        type=int,
        default=32,
        help="Cons-DecAF: K bins for global soft label e_g and estimator logits_hat (>=2).",
    )
    parser.add_argument(
        "--cons_mac_ce_coef",
        type=float,
        default=1.0,
        help="Weight on L_CE (KL(e_g || softmax(hat_e))); 0 disables CE auxiliary (separate ce_optimizer).",
    )
    parser.add_argument(
        "--cons_mac_ce_lr",
        type=float,
        default=-1.0,
        help="Learning rate for CE stack; <0 uses --lr.",
    )
    parser.add_argument(
        "--cons_mac_distill_coef",
        type=float,
        default=0.0,
        help="Policy distillation: weight on MSE(student logits, teacher logits) + MSE(h, teacher h); 0=off.",
    )
    parser.add_argument(
        "--cons_mac_teacher_model_dir",
        type=str,
        default="",
        help="Path to teacher actor state_dict (.pt) for distillation; empty=none.",
    )
    parser.add_argument(
        "--cons_decaf_omega_f",
        type=float,
        default=1.0,
        help="Paper ω_f: weight on formation reward r_f (HD-based).",
    )
    parser.add_argument(
        "--cons_decaf_omega_v",
        type=float,
        default=1.0,
        help="Paper ω_v: weight on navigation reward r_v (centroid to destination).",
    )
    parser.add_argument(
        "--cons_decaf_omega_c",
        type=float,
        default=1.0,
        help="Paper ω_c: weight on collision-count penalty r_c.",
    )
    parser.add_argument(
        "--cons_decaf_omega1_lag",
        type=float,
        default=0.1,
        help="Paper ω1: lag coefficient on previous step r_f in formation reward.",
    )
    parser.add_argument(
        "--cons_decaf_omega2_lag",
        type=float,
        default=0.1,
        help="Paper ω2: lag coefficient on previous step r_v in navigation reward.",
    )
    parser.add_argument(
        "--cons_decaf_delta_safe",
        type=float,
        default=0.0,
        help="Paper δ_safe for pairwise collision count; 0 uses 2*robot_radius.",
    )
    parser.add_argument(
        "--cons_decaf_goal_disk_coef",
        type=float,
        default=0.0,
        help="If >0, add nd_discount_goal * goal_disk (per agent) on top of shared paper team reward.",
    )
    parser.add_argument(
        "--dynamic_same_target_conflict_dist",
        type=float,
        default=2.0,
        help="Same target_id and pairwise distance below this triggers shaping penalty.",
    )
    parser.add_argument("--dynamic_path_reward_scale", type=float, default=1.0, help="Scale for -sum(step distances) team term.")
    parser.add_argument(
        "--dynamic_same_target_penalty_scale",
        type=float,
        default=3.6,
        help="Scale for same-target proximity penalty (per pair, split across agents).",
    )
    parser.add_argument(
        "--dynamic_target_switch_penalty",
        type=float,
        default=1.48,
        help="Base penalty when an agent changes target_id (per step, on the switching agent).",
    )
    parser.add_argument(
        "--dynamic_formation_success_bonus",
        type=float,
        default=50.0,
        help="Per-agent bonus when all claimed distinct goals with no episode collision.",
    )
    parser.add_argument(
        "--dynamic_discount_formation",
        type=float,
        default=1.0,
        help="Scale on r_formation_raw (-sqrt(Laplacian error)) in dynamic mode; 0 disables. "
        "Multiplied by formation_time_weight_* schedule over episode.",
    )
    parser.add_argument(
        "--dynamic_team_dist_progress_scale",
        type=float,
        default=0.175,
        help="Shared shaping: reward drop in sum of distances-to-assigned-goals (÷n per agent). Encourages coordinated assignment.",
    )
    parser.add_argument(
        "--dynamic_target_overcommit_scale",
        type=float,
        default=0.88,
        help="Shared penalty for extra agents (beyond one) choosing the same target before success (÷n).",
    )
    parser.add_argument(
        "--dynamic_arrival_reward",
        type=float,
        default=28.0,
        help="Per-agent bonus the step a robot successfully claims its goal (ensures reaching is reinforced).",
    )
    parser.add_argument(
        "--dynamic_proximity_reward_scale",
        type=float,
        default=0.15,
        help="Per-agent exp(-dist/sigma) toward current goal while not yet successful (0 to disable).",
    )
    parser.add_argument(
        "--dynamic_proximity_sigma",
        type=float,
        default=8.0,
        help="Distance scale for dynamic_proximity_reward.",
    )
    parser.add_argument(
        "--dynamic_arrival_require_outside_entry",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: arrival bonus only if previous position was outside the claimed goal radius (anti spam). 0: legacy.",
    )
    parser.add_argument(
        "--dynamic_zero_nav_on_target_switch",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: zero distance-progress r_nav when target_id changed this step (gx,gy jump makes delta misleading). 0: off.",
    )
    parser.add_argument(
        "--dynamic_nav_scale_on_target_switch",
        type=float,
        default=0.0,
        help="If dynamic_zero_nav_on_target_switch=0, multiply r_nav by this when switched (else ignored).",
    )
    parser.add_argument(
        "--dynamic_prox_scale_on_target_switch",
        type=float,
        default=0.12,
        help="Multiply proximity shaping when target switched this step (reduces reward churn near goals). Use 0 to zero.",
    )
    parser.add_argument(
        "--dynamic_switch_near_goal_extra",
        type=float,
        default=0.85,
        help="Added to switch penalty when robot is within margin of any goal center (discourages in-zone target flicker). 0 disables extra.",
    )
    parser.add_argument(
        "--dynamic_switch_near_goal_margin_radius",
        type=float,
        default=1.75,
        help="Margin = this × robot_radius: if min distance to any goal center ≤ margin, switch_near_goal_extra applies.",
    )
    parser.add_argument(
        "--dynamic_crowding_dist",
        type=float,
        default=3.0,
        help="Pairwise distance below this (world units) builds crowding penalty among non-done robots. 0 disables.",
    )
    parser.add_argument(
        "--dynamic_crowding_penalty_scale",
        type=float,
        default=0.24,
        help="Shared term: subtract scale × Σ_{i<j} w_ij² / n with w_ij=max(0,1-d_ij/crowding_dist). Encourages spreading out.",
    )
    parser.add_argument(
        "--dynamic_cluster_same_target_boost",
        type=float,
        default=1.35,
        help="Multiply same-target conflict penalty for tight pairs: if crowding_dist>0, only when d<crowding_dist; "
        "if crowding_dist=0, applies to all same-target conflicts (d<conflict_dist). 1=no boost.",
    )
    parser.add_argument(
        "--dynamic_explore_undervisible_scale",
        type=float,
        default=0.38,
        help="Slots dynamic obs only: reward moving toward nearest M-packed goal outside R_vis (heading × speed). "
        "Scaled by local teammate density (dense→full, sparse→×dynamic_explore_low_density_mult). 0=off.",
    )
    parser.add_argument(
        "--dynamic_explore_v_ref",
        type=float,
        default=1.0,
        help="Caps speed factor in explore bonus at min(v/v_ref, 1).",
    )
    parser.add_argument(
        "--dynamic_explore_min_shortfall",
        type=int,
        default=1,
        help="Require at least this many of the M packed nearest goals to be outside R_vis before explore bonus applies.",
    )
    parser.add_argument(
        "--dynamic_explore_require_unclaimed_outside",
        type=int,
        default=0,
        choices=[0, 1],
        help="1: only steer toward outside-R_vis goals in the M-pack that are still unclaimed (claimed_by=-1).",
    )
    parser.add_argument(
        "--dynamic_local_density_radius",
        type=float,
        default=0.0,
        help="Count active teammates within this distance for local sparse/dense gating; 0 uses dynamic_crowding_dist "
        "(or 3.0 if that is 0).",
    )
    parser.add_argument(
        "--dynamic_explore_density_sparse_max",
        type=int,
        default=1,
        help="If local neighbor count ≤ this, scale explore_undervisible by dynamic_explore_low_density_mult (sparse: "
        "focus on reaching current target via urgency, not exploring away).",
    )
    parser.add_argument(
        "--dynamic_explore_density_dense_min",
        type=int,
        default=3,
        help="If local neighbor count ≥ this, explore_undervisible uses full scale (dense: push outer agents toward "
        "outside-R_vis targets).",
    )
    parser.add_argument(
        "--dynamic_explore_low_density_mult",
        type=float,
        default=0.12,
        help="Min multiplier on explore_undervisible when locally sparse (see explore_density_*).",
    )
    parser.add_argument(
        "--dynamic_shaping_commit_rvis_mult",
        type=float,
        default=1.0,
        help="When dist2goal to the assigned target ≤ this × dynamic_target_vis_radius, turn off exploration shaping "
        "that points at other goals (undervisible explore, low-density flee unless chasing a claimed target, "
        "CTDE nearest-unclaimed progress, cluster-v2 explore role). 0 disables the commit zone.",
    )
    parser.add_argument(
        "--dynamic_sparse_urgency_max_mult",
        type=float,
        default=1.32,
        help="Max multiplier on r_nav and r_prox when locally sparse, only if not losing a same-target race to a closer "
        "neighbor (see sparse_urgency_only_when_uncontested).",
    )
    parser.add_argument(
        "--dynamic_sparse_urgency_neighbors_max",
        type=int,
        default=1,
        help="If local neighbor count ≤ this, apply sparse_urgency_max_mult to nav/prox (linear blend to 1.0).",
    )
    parser.add_argument(
        "--dynamic_sparse_urgency_neighbors_norm",
        type=int,
        default=4,
        help="If local neighbor count ≥ this, urgency multiplier is 1.0 (between this and sparse_max, linear).",
    )
    parser.add_argument(
        "--dynamic_claimed_target_penalty_scale",
        type=float,
        default=0.62,
        help="Per-step penalty (× nd_discount_nav) while target_id points to a goal already claimed by another agent. "
        "0=off.",
    )
    parser.add_argument(
        "--dynamic_ctde_remaining_target_shaping_scale",
        type=float,
        default=0.28,
        help="CTDE-oriented training bonus: progress toward the nearest *unclaimed* goal (global claimed_by). "
        "Near → +reward, away → penalty via (pre_dist - dist)×coef. 0=off.",
    )
    parser.add_argument(
        "--dynamic_ctde_remaining_target_progress_coef",
        type=float,
        default=5.0,
        help="Multiplies (pre_dist_nearest_unclaimed - dist_nearest_unclaimed) before shaping_scale.",
    )
    parser.add_argument(
        "--dynamic_ctde_remaining_shaping_require_centralized_v",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: only apply remaining-target shaping when --use_centralized_V (CTDE critic). 0: always apply when scale>0.",
    )
    parser.add_argument(
        "--dynamic_sparse_contest_margin",
        type=float,
        default=0.18,
        help="Local contest: another agent within local_density_radius with the same target_id and closer to that goal "
        "by more than this margin (world units) counts as stronger competition → no sparse urgency bonus.",
    )
    parser.add_argument(
        "--dynamic_sparse_urgency_only_when_uncontested",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: apply sparse nav/prox urgency only when not locally contested on current target (see sparse_contest_margin).",
    )
    parser.add_argument(
        "--dynamic_low_density_explore_scale",
        type=float,
        default=0.36,
        help="When locally contested on same target OR chasing another agent's claimed goal, reward heading toward "
        "lower agent density (away from neighbor centroid, or sparsest angular sector). 0=off.",
    )
    parser.add_argument(
        "--dynamic_low_density_explore_v_ref",
        type=float,
        default=1.0,
        help="Caps low-density explore bonus with min(v/v_ref, 1).",
    )
    parser.add_argument(
        "--dynamic_low_density_sector_radius",
        type=float,
        default=0.0,
        help="Radius for counting agents in sparsest-sector heuristic; 0 → max(2×local_density_radius, 8).",
    )
    parser.add_argument(
        "--dynamic_low_density_sector_cos",
        type=float,
        default=0.707,
        help="Cosine threshold for forward wedge (e.g. 0.707 ≈ 45° half-angle) when scoring sector occupancy.",
    )
    parser.add_argument(
        "--dynamic_low_density_sector_count",
        type=int,
        default=8,
        help="Number of angular bins for sparsest-direction search.",
    )
    parser.add_argument(
        "--dynamic_reciprocal_swap_reward_scale",
        type=float,
        default=0.55,
        help="Per-agent bonus when this step it swaps targets with another agent (each takes the other's previous "
        "target_id) while both move toward each other (heading). 0=off.",
    )
    parser.add_argument(
        "--dynamic_reciprocal_swap_switch_penalty_mult",
        type=float,
        default=0.45,
        help="Multiply target-switch penalty by this when the step is a reciprocal swap (after other switch mods). "
        "1.0=no relief.",
    )
    parser.add_argument(
        "--dynamic_swap_approach_cos_thresh",
        type=float,
        default=0.12,
        help="Reciprocal swap bonus requires both agents' headings to point toward each other: "
        "dot(heading_i, r_j-r_i)/||r_j-r_i|| and symmetric j must exceed this (cos ~ 0.12 ≈ 83° cone).",
    )
    parser.add_argument(
        "--dynamic_goal_contention_penalty_scale",
        type=float,
        default=0.0,
        help="Optional legacy shared pile-up penalty near goal centers. Default 0: use dynamic_claimed_target_penalty "
        "and density-gated explore instead.",
    )
    parser.add_argument(
        "--dynamic_goal_contention_radius",
        type=float,
        default=0.0,
        help="World distance to goal center for contention count; 0 → max(2×robot_radius, 0.5).",
    )
    parser.add_argument(
        "--dynamic_switch_penalty_relief_when_team_closer",
        type=float,
        default=1.0,
        help="Multiply switch penalty by this when team total distance decreases that step; 1.0 = no relief (discourage churn).",
    )
    parser.add_argument(
        "--dynamic_switch_low_hold_boost_scale",
        type=float,
        default=3.2,
        help="Extra switch penalty multiplier ~ (1 + scale/(1+hold_steps)) where hold_steps is time on previous target before switch; discourages rapid target oscillation.",
    )
    # CTDE cluster reward v2 (global clustering + roles; requires centralized V unless overridden)
    parser.add_argument(
        "--dynamic_reward_cluster_v2",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: use spatial-cluster roles + stronger penalties (see dynamic_spatial_cluster_*). 0: legacy dynamic reward mix.",
    )
    parser.add_argument(
        "--dynamic_cluster_reward_requires_centralized_v",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: cluster v2 only when --use_centralized_V. 0: cluster v2 always when dynamic_reward_cluster_v2=1.",
    )
    parser.add_argument(
        "--dynamic_spatial_cluster_link_dist",
        type=float,
        default=4.0,
        help="Agents within this distance (active only) merge into one spatial cluster.",
    )
    parser.add_argument(
        "--dynamic_cluster_target_neighborhood_radius",
        type=float,
        default=10.0,
        help="Unclaimed goal counts as 'near cluster' if within this radius of cluster centroid.",
    )
    parser.add_argument(
        "--dynamic_cluster_unclaimed_quota_match_agents",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: 'enough' nearby unclaimed targets iff count >= active agents in cluster. 0: use min_quota only.",
    )
    parser.add_argument(
        "--dynamic_cluster_unclaimed_near_min_quota",
        type=int,
        default=2,
        help="When quota_match_agents=0, need at least this many unclaimed goals near cluster.",
    )
    parser.add_argument(
        "--dynamic_cluster_far_agent_fraction",
        type=float,
        default=0.35,
        help="When nearby unclaimed are insufficient, this fraction (ceil) of cluster is explore role: agents closest to unclaimed targets outside the cluster neighborhood head for those; if none exist outside, selection falls back to farthest-from-nearest-unclaimed.",
    )
    parser.add_argument(
        "--dynamic_cluster_local_unc_shaping_scale",
        type=float,
        default=0.44,
        help="v2: progress toward nearest unclaimed goal near cluster centroid (× progress_coef).",
    )
    parser.add_argument(
        "--dynamic_cluster_explore_shaping_scale",
        type=float,
        default=0.4,
        help="v2: explorers progress toward global nearest unclaimed goal.",
    )
    parser.add_argument(
        "--dynamic_cluster_explorer_no_target_penalty",
        type=float,
        default=0.58,
        help="v2: per-step penalty for explorer role when no unclaimed goal remains anywhere.",
    )
    parser.add_argument(
        "--dynamic_cluster_progress_coef",
        type=float,
        default=5.0,
        help="v2: multiplies (pre_dist - dist) for local_unc / explore shaping.",
    )
    parser.add_argument(
        "--dynamic_cluster_rest_nav_prox_mult",
        type=float,
        default=1.28,
        help="v2: multiply r_nav and r_prox for 'rest' cluster agents (occupy nearest assigned target).",
    )
    parser.add_argument(
        "--dynamic_cluster_rest_dispersion_scale",
        type=float,
        default=0.14,
        help="v2: penalty scale exp(-d/sigma) sum over cluster mates (encourage sparse spread).",
    )
    parser.add_argument(
        "--dynamic_cluster_rest_dispersion_sigma",
        type=float,
        default=4.0,
        help="v2: length scale for rest-role dispersion penalty.",
    )
    parser.add_argument(
        "--dynamic_v2_shared_conflict_mult",
        type=float,
        default=1.55,
        help="v2: multiply same-target conflict penalty weight.",
    )
    parser.add_argument(
        "--dynamic_v2_shared_overcommit_mult",
        type=float,
        default=1.75,
        help="v2: multiply overcommit penalty scale.",
    )
    parser.add_argument(
        "--dynamic_v2_collision_hard_penalty",
        type=float,
        default=-95.0,
        help="v2: r_avoid when robot.collision (replaces -60).",
    )
    parser.add_argument(
        "--dynamic_v2_dmin_avoid_mult",
        type=float,
        default=1.5,
        help="v2: multiply soft dmin avoidance exp term.",
    )
    parser.add_argument(
        "--dynamic_formation_time_invariant",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: dynamic formation Laplacian term ignores formation_time_weight schedule (constant 1.0).",
    )
    parser.add_argument(
        "--dynamic_loiter_goal_dist_thresh",
        type=float,
        default=2.5,
        help="Distance to current gx,gy below this accumulates loiter steps (for switch-after-loiter penalty).",
    )
    parser.add_argument(
        "--dynamic_loiter_steps_for_switch_penalty",
        type=int,
        default=6,
        help="If agent switches target after at least this many consecutive near-goal steps, add extra switch penalty.",
    )
    parser.add_argument(
        "--dynamic_switch_after_loiter_extra",
        type=float,
        default=1.7,
        help="Added to switch penalty when loiter threshold exceeded at switch.",
    )
    parser.add_argument(
        "--dynamic_switch_sparse_neighbor_max",
        type=int,
        default=2,
        help="If local neighbor count ≤ this and agent switches target, add sparse churn penalty.",
    )
    parser.add_argument(
        "--dynamic_switch_sparse_extra",
        type=float,
        default=1.2,
        help="Extra switch penalty when locally sparse but still churning targets.",
    )
    parser.add_argument(
        "--dynamic_switch_episode_prior_coef",
        type=float,
        default=0.12,
        help="Add this × (number of prior target switches this episode) to switch penalty when switching "
        "(0=off). First switch in episode adds 0; ramps up for chronic churn.",
    )
    parser.add_argument(
        "--dynamic_v2_disable_reciprocal_swap_bonus",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: v2 mode zeroes reciprocal-swap reward (keep switch relief mult if desired).",
    )
    # Local target reasoning (e.g. fixed-M slots / possi): M and R_vis are separate knobs; defaults follow
    # the same neighbourhood hyperparameters as agent_state_mode (not hard-coded literals).
    parser.add_argument(
        "--dynamic_target_slot_count",
        type=int,
        default=None,
        help="Number M of local target slots. If unset, uses neighbor_n (capped by K=num_goal_targets in env).",
    )
    parser.add_argument(
        "--dynamic_target_vis_radius",
        type=float,
        default=None,
        help="Visibility radius R_vis for per-target features (e.g. possi). If unset, uses neighbor_radius.",
    )
    parser.add_argument(
        "--disable_neighbor_attn_lstm_actor",
        action="store_true",
        default=False,
        help="When using dynamic goals, disable nearest-teammate self-attn+LSTM actor branch (legacy obs tail).",
    )
    parser.add_argument(
        "--actor_neighbor_n",
        type=int,
        default=None,
        help="How many nearest teammates to encode for the actor neighbor branch; default = neighbor_n.",
    )
    # Attention + intent communication actor (fixed / pre-assigned targets only; incompatible with dynamic goals)
    parser.add_argument(
        "--use_attn_comm_actor",
        action="store_true",
        default=False,
        help="Use grouped GAT-style encoders + intent message attention on robot obs (requires fixed targets).",
    )
    parser.add_argument(
        "--attn_comm_radius",
        type=float,
        default=None,
        help="Only pack allies/humans within this distance (world units). If unset, uses neighbor_radius.",
    )
    parser.add_argument(
        "--attn_comm_ally_slots",
        type=int,
        default=None,
        help="Max teammate slots in obs (fixed width). If unset, uses neighbor_n.",
    )
    parser.add_argument(
        "--attn_comm_human_slots",
        type=int,
        default=None,
        help="Max pedestrian slots in obs. If unset, uses neighbor_n.",
    )
    parser.add_argument(
        "--attn_comm_message_dim",
        type=int,
        default=16,
        help="Intent / broadcast vector width (also extra obs per ally slot). Smaller = lighter.",
    )
    parser.add_argument(
        "--attn_comm_hidden_dim",
        type=int,
        default=0,
        help="Internal AttnComm trunk width (embeddings + attention). 0 = use --hidden_size (widest). "
        "E.g. 64 with hidden_size 128 shrinks MHA/MLP params and VRAM; prefer even values divisible by heads.",
    )
    parser.add_argument(
        "--attn_comm_gat_heads",
        type=int,
        default=2,
        help="Multi-head attention heads (attn_comm_hidden_dim or hidden_size must be divisible).",
    )
    parser.add_argument(
        "--attn_comm_state_dim",
        type=int,
        default=None,
        help="GRU hidden dim for long-term comm state; default = hidden_size, or = attn_comm_hidden_dim when that is set > 0.",
    )
    parser.add_argument(
        "--attn_comm_reward_coef",
        type=float,
        default=0.0,
        help="Scale for optional comm shaping (0=off). Use with use_attn_comm_actor; fixed-target mode only.",
    )
    parser.add_argument(
        "--attn_comm_reward_w_align",
        type=float,
        default=1.0,
        help="Weight: mean cosine similarity of my broadcast with in-radius teammates' broadcasts.",
    )
    parser.add_argument(
        "--attn_comm_reward_w_diversity",
        type=float,
        default=0.25,
        help="Weight: mean pairwise L2 distance of normalized broadcasts (team-level, discourages collapse).",
    )
    parser.add_argument(
        "--attn_comm_reward_w_smooth",
        type=float,
        default=0.08,
        help="Weight: negative mean squared step change in broadcast vs previous step (per agent).",
    )
    parser.add_argument(
        "--attn_comm_max_ppo_samples_per_gpu",
        type=int,
        default=512,
        help="use_attn_comm_actor: raise num_mini_batch if needed so each PPO minibatch has at most this "
        "many samples (episode_length * n_rollout_threads * num_agents total). Lowers VRAM spikes; 0 disables.",
    )

    # Non-dynamic mode: denser shaping for decentralized / local-obs goal reaching (ignored when dynamic is on)
    parser.add_argument(
        "--nd_goal_progress_coef",
        type=float,
        default=8.5,
        help="Scales (clipped) distance progress before discount_nav.",
    )
    parser.add_argument(
        "--nd_nav_progress_clip",
        type=float,
        default=0.42,
        help="Max absolute per-step (pre_dist2goal-dist2goal) before × coef; 0 disables clipping.",
    )
    parser.add_argument(
        "--nd_timeout_no_goal_penalty",
        type=float,
        default=-7.0,
        help="Raw r_nav term on the timeout step if agent is not at goal and not in collision (≤0 typical). 0=off.",
    )
    parser.add_argument(
        "--nd_proximity_reward_scale",
        type=float,
        default=0.25,
        help="Adds scale*exp(-dist2goal/sigma) each step (0 to disable).",
    )
    parser.add_argument(
        "--nd_proximity_sigma",
        type=float,
        default=10.0,
        help="Distance scale for nd_proximity_reward (larger = reward extends farther).",
    )
    parser.add_argument(
        "--nd_heading_reward_scale",
        type=float,
        default=0.2,
        help="Bonus for heading aligned with goal vector, scaled by normalized speed (0 to disable).",
    )
    parser.add_argument(
        "--nd_heading_v_ref",
        type=float,
        default=1.0,
        help="Speed normalization for nd_heading_reward.",
    )
    parser.add_argument(
        "--nd_arrival_reward",
        type=float,
        default=9.0,
        help="Extra r_goal when inside goal disk each step; still multiplied by discount_goal.",
    )
    parser.add_argument(
        "--nd_goal_stay_reward",
        type=float,
        default=4.5,
        help="Additional r_goal per step while inside goal disk (on top of nd_arrival_reward). 0 = off.",
    )
    parser.add_argument(
        "--nd_goal_leave_penalty",
        type=float,
        default=36.0,
        help="Subtracted from r_goal when agent was inside the goal disk last state and is outside now (strong discourages leaving).",
    )
    parser.add_argument(
        "--nd_goal_inside_velocity_bonus",
        type=float,
        default=10.0,
        help="Added to r_goal each step inside goal when speed v > 0 (replaces former hardcoded +5).",
    )
    parser.add_argument(
        "--nd_goal_terminal_reward",
        type=float,
        default=15.0,
        help="One-shot bonus the first step the agent enters the goal disk (adds to weighted reward). "
        "When >0, staying inside still receives shaped reward (not zeroed). 0 = legacy.",
    )
    parser.add_argument(
        "--nd_discount_formation",
        type=float,
        default=0.0,
        help="Base weight on r_formation_raw (always <=0 raw: Laplacian mismatch penalty only). "
        "Effective scale = nd_discount_formation * formation_time_weight_* schedule. 0 = off.",
    )
    parser.add_argument(
        "--formation_time_weight_start",
        type=float,
        default=1.0,
        help="Multiplier on formation weight at episode start (tau=0).",
    )
    parser.add_argument(
        "--formation_time_weight_end",
        type=float,
        default=0.65,
        help="Multiplier on formation weight when tau_eff reaches 1. Higher = formation penalty stays stronger late.",
    )
    parser.add_argument(
        "--formation_time_weight_decay_horizon",
        type=float,
        default=0.88,
        help="τ_eff = min(1, τ/horizon) with τ = t_elapsed/time_limit. Larger = slower decay toward end weight "
        "(formation penalty fades more slowly). 1.0 = linear over full episode.",
    )
    parser.add_argument(
        "--nd_discount_avoid",
        type=float,
        default=28.0,
        help="Weight on r_avoid_raw (soft discomfort uses exp penalty; lower reduces -40~-50 spikes vs nav).",
    )
    parser.add_argument(
        "--nd_discount_nav",
        type=float,
        default=28.0,
        help="Weight on r_nav_raw (distance progress + optional proximity/heading).",
    )
    parser.add_argument(
        "--nd_discount_goal",
        type=float,
        default=290.0,
        help="Weight on r_goal_raw (arrival/stay/leave shaping inside goal disk).",
    )

    parser.add_argument(
        "--randomize_robot_initial_positions",
        action="store_true",
        default=True,
        help="Sample robot starts in a box with minimum separation (dynamic and non-dynamic mode).",
    )
    parser.add_argument(
        "--robot_initial_spawn_mode",
        type=str,
        default="random_box",
        choices=("random_box", "cluster_comm", "cluster_disk"),
        help="When randomize_robot_initial_positions is True: random_box=i.i.d. in the init rectangle; "
        "cluster_comm=regular polygon on a ring (legacy); cluster_disk=uniform in a disk (rejection sampling) "
        "using the same reference radius as cluster_comm (see robot_init_cluster_*).",
    )
    parser.add_argument(
        "--robot_init_cluster_radius_mode",
        type=str,
        default="comm",
        choices=("comm", "comm_vis_adaptive"),
        help="cluster_comm reference radius when --robot_init_cluster_comm_radius is unset: "
        "comm=min of positive attn_comm_radius and undetermined_comm_radius (legacy). "
        "comm_vis_adaptive=min of all positive among attn radius, undetermined comm, undetermined_obs_goal_radius, "
        "dynamic_target_vis_radius (if dynamic goals), neighbor_radius (if agent_state_mode=nearest_n_radius).",
    )
    parser.add_argument(
        "--robot_init_cluster_comm_radius",
        type=float,
        default=None,
        help="If set (>0), overrides cluster reference radius for cluster_comm/cluster_disk. Otherwise see "
        "--robot_init_cluster_radius_mode.",
    )
    parser.add_argument("--robot_init_x_min", type=float, default=-8.0)
    parser.add_argument("--robot_init_x_max", type=float, default=8.0)
    parser.add_argument("--robot_init_y_min", type=float, default=-8.0)
    parser.add_argument("--robot_init_y_max", type=float, default=8.0)
    parser.add_argument(
        "--robot_init_min_separation_margin",
        type=float,
        default=0.15,
        help="Margin beyond r_i+r_j for collision-free init sampling.",
    )

    parser.add_argument(
        "--use_human_obs",
        action="store_false",
        default=True,
        help="Whether to use human observations in actor/critic inputs (default: True). If set, disable human obs usage.",
    )

    # agent neighbourhood / state aggregation parameters
    # agent_state_mode:
    #   - "all":         use all agents' states (current default behaviour)
    #   - "nearest_n":   use the nearest `neighbor_n` agents (by distance)
    #   - "nearest_n_radius": use up to `neighbor_n` nearest agents within `neighbor_radius`,
    #                         and pad remaining slots with `neighbor_padding_value`
    parser.add_argument(
        "--agent_state_mode",
        type=str,
        default="nearest_n_radius",
        choices=["all", "nearest_n", "nearest_n_radius"],
        help="How to construct agent-centric inputs: all agents, nearest N, or nearest N within a radius.",
    )
    parser.add_argument(
        "--neighbor_n",
        type=int,
        default=10,
        help="Number of neighbour agents to use when agent_state_mode is 'nearest_n' or 'nearest_n_radius'.",
    )
    parser.add_argument(
        "--neighbor_radius",
        type=float,
        default=5.0,
        help="Neighbour radius (in position space) when agent_state_mode is 'nearest_n_radius'.",
    )
    parser.add_argument(
        "--neighbor_distance_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan"],
        help="Distance metric for neighbour selection: euclidean or manhattan.",
    )
    parser.add_argument(
        "--neighbor_padding_value",
        type=float,
        default=0.0,
        help="Padding value used to fill unused neighbour slots when there are fewer than neighbour_n agents.",
    )

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=100, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy",
        action="store_false",
        default=True,
        help="Whether agent share the same policy",
    )
    parser.add_argument("--use_centralized_V",
        action="store_false",
        default=True,
        help="Whether to use centralized V function",
    )
    parser.add_argument("--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument("--use_stacked_frames",
        action="store_true",
        default=False,
        help="Whether to use stacked_frames",
    )
    parser.add_argument("--hidden_size",type=int,default=128,help="Dimension of hidden layers for actor/critic networks",)
    parser.add_argument("--layer_N",type=int,default=1,help="Number of layers for actor/critic networks",)
    parser.add_argument("--use_ReLU", action="store_false", default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument("--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    parser.add_argument("--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument("--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy",action="store_true",default=False,help="Whether to use a naive recurrent policy",)
    parser.add_argument("--use_recurrent_policy",action="store_false",default=False,help="use a recurrent policy",)
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length",type=int,default=10,help="Time length of chunks used to train a recurrent_policy",)

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate (default: 4e-4)")
    parser.add_argument("--critic_lr",type=float,default=4e-4,help="critic learning rate (default: 4e-4)",)
    parser.add_argument("--opti_eps",type=float,default=1e-5,help="RMSprop optimizer epsilon (default: 1e-5)",)
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=12, help="number of ppo-update epochs (default: 12)")
    parser.add_argument("--use_clipped_value_loss",action="store_false",default=True,
                        help="by default, clip loss value. If set, do not clip loss value.",)
    parser.add_argument("--clip_param",type=float,default=0.2,help="ppo clip parameter (default: 0.2)",)
    parser.add_argument("--num_mini_batch",type=int,default=100,help="number of batches for ppo (default: 1)",)
    parser.add_argument("--entropy_coef",type=float,default=0.02,help="entropy term coefficient (default: 0.02)",)
    parser.add_argument("--value_loss_coef",type=float,default=1,help="value loss coefficient (default: 0.5)",)
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="max norm of gradients (default: 5)",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="time duration between contiunous twice log printing.",
    )
    parser.add_argument(
        "--save_reward_terms",
        action="store_true",
        default=False,
        help="If set, append per-step reward breakdown to logs/reward_terms.csv on the same episodes as TensorBoard (log_interval); use reward_terms_log_stride to subsample steps.",
    )
    parser.add_argument(
        "--reward_terms_max_envs",
        type=int,
        default=1,
        help="When save_reward_terms: only log this many parallel envs (indices 0..N-1).",
    )
    parser.add_argument(
        "--reward_terms_log_stride",
        type=int,
        default=1,
        help="When save_reward_terms: log every k-th env step (1 = every step).",
    )

    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="number of episodes of a single evaluation.",
    )

    # render parameters
    parser.add_argument("--save_gifs",action="store_true",default=False,help="by default, do not save render video. If set, save video.",)
    parser.add_argument("--use_render",action="store_true",default=False,
        help="by default, do not render the env during training. If set, start render. \
        Note: something, the environment has internal render process which is not controlled by this hyperparam.",)
    parser.add_argument("--render_episodes",type=int,default=2,help="the number of episodes to render a given env",)
    parser.add_argument("--visualize",type=bool,default=False,help='whether to use a visual interface')
    parser.add_argument("--ifi",type=float,default=0.1,help="the play interval of each rendered image in saved video.",)
    parser.add_argument("--method", type=str, default='ppo', help="ppo, orca, apf")

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Pretrained shared policy: directory containing actor.pt (and critic.pt for training resume), "
        "or the same directory with 4.pt if actor.pt is absent, or a direct path to a *.pt actor file.",
    )

    # agent parameters
    parser.add_argument("--num_humans", type=int, default=0, help="number of dynamic obstacles")
    parser.add_argument("--num_attention_agents", type=int, default=10, help="number of agents that should be paid attention")
    parser.add_argument("--for_edge", type=int,default=2, help='the formation edge lenth')
    parser.add_argument("--robot_radius", type=float,default=0.3, help='the radius of robot')
    parser.add_argument("--human_radius", type=float,default=0.3, help='the radius of human')
    parser.add_argument("--dcf_dist",type=float,default=0.2,help='discomfort distance of robot and human')
    parser.add_argument("--base_v",type=float,default=0.25,help='When use discrete envirnment,the base velosity in action space')
    parser.add_argument("--randomize_attributes",type=bool,default=False,help='Randomize humans radius and preferred speed')
    parser.add_argument("--human_action",type=str,default='square_crossing',
                        help='human(dynamic obstacle) act trajectory,include square_crossing,circle_crossing,mixed')

    # Font pattern (agent formation target) selection
    # - training: length count is unique (single int)
    # - render: length count can be multiple (list)
    parser.add_argument(
        "--train_font_pattern_length",
        type=int,
        default=10,
        help="Font pattern length used for training (must be a single value matching dataset file name).",
    )
    parser.add_argument(
        "--train_font_pattern_policy",
        type=str,
        default="all",
        choices=["all", "only", "must_contain"],
        help="Training pattern pool policy: use all patterns, only the listed ones, or must include the listed ones.",
    )
    parser.add_argument(
        "--train_font_pattern_names",
        type=str,
        default="",
        help="Comma-separated pattern names for training (used by 'only'/'must_contain' policies).",
    )
    parser.add_argument(
        "--train_font_pattern_allow_repeat",
        action="store_true",
        default=False,
        help="When n_env_threads <= n_patterns, allow repeating patterns across different env threads.",
    )
    parser.add_argument(
        "--render_font_pattern_lengths",
        type=str,
        default="10",
        help="Comma-separated font pattern lengths for render (e.g. '10' or '10,11').",
    )
    parser.add_argument(
        "--render_font_pattern_policy",
        type=str,
        default="only",
        choices=["all", "only", "must_contain"],
        help="Render pattern pool policy: use all patterns, only the listed ones, or must include the listed ones.",
    )
    parser.add_argument(
        "--render_font_pattern_names",
        type=str,
        default="S",
        help="Comma-separated pattern names for render (used by 'only'/'must_contain' policies).",
    )
    parser.add_argument("--square_width",type=float,default=10,help='if select square_crossing in human_action,the width of the square')
    parser.add_argument("--circle_radius",type=float,default=10,help='if select circle_crossing in human_action,the radius of the circle')

    # orca parameters
    parser.add_argument("--max_speed", type=float, default=1.25)
    parser.add_argument("--time_horizon",type=int,default=5)
    parser.add_argument("--time_horizon_obst",type=int,default=5)
    parser.add_argument("--neighbor_dist",type=int,default=10)
    parser.add_argument("--max_neighbors",type=int,default=10)
    parser.add_argument("--v_pref",type=float,default=1,help='the preferred velocity of agents')


    return parser


def resolve_attn_comm_args(args):
    """
    Fill attn_comm_radius / *_slots from neighbour geometry when unset (no separate hard-coded defaults).
    """
    if not getattr(args, "use_attn_comm_actor", False):
        return args
    nr = float(getattr(args, "neighbor_radius", 5.0))
    nn = int(getattr(args, "neighbor_n", 10))
    if getattr(args, "attn_comm_radius", None) is None:
        args.attn_comm_radius = nr
    if getattr(args, "attn_comm_ally_slots", None) is None:
        args.attn_comm_ally_slots = nn
    if getattr(args, "attn_comm_human_slots", None) is None:
        args.attn_comm_human_slots = nn
    hd = int(getattr(args, "attn_comm_hidden_dim", 0) or 0)
    if hd > 0 and getattr(args, "attn_comm_state_dim", None) is None:
        args.attn_comm_state_dim = hd
    return args


def resolve_attn_comm_ppo_batch_args(args):
    """
    AttnComm actor + critic forward are memory-heavy. Increase num_mini_batch so
    mini_batch_size = (T * n_env * n_agents) / num_mini_batch stays bounded.
    """
    import math

    if not getattr(args, "use_attn_comm_actor", False):
        return args
    cap = int(getattr(args, "attn_comm_max_ppo_samples_per_gpu", 0) or 0)
    if cap <= 0:
        return args
    T = int(getattr(args, "episode_length", 1))
    N = int(getattr(args, "n_rollout_threads", 1))
    M = int(getattr(args, "num_agents", 1))
    batch_size = max(1, T * N * M)
    if batch_size <= cap:
        return args
    need_mb = int(math.ceil(batch_size / float(cap)))
    cur = max(1, int(getattr(args, "num_mini_batch", 1)))
    if need_mb > cur:
        old = cur
        args.num_mini_batch = need_mb
        print(
            f"[attn_comm] num_mini_batch {old} -> {need_mb} "
            f"(rollout batch_size={batch_size}, target <= {cap} samples per PPO minibatch)"
        )
    return args


def resolve_dynamic_target_reasoning_args(args):
    """
    Fill dynamic_target_slot_count / dynamic_target_vis_radius when left unset so they track
    neighbour geometry (neighbor_n, neighbor_radius) instead of ad-hoc constants.
    """
    if getattr(args, "dynamic_target_slot_count", None) is None:
        args.dynamic_target_slot_count = int(getattr(args, "neighbor_n", 10))
    if getattr(args, "dynamic_target_vis_radius", None) is None:
        args.dynamic_target_vis_radius = float(getattr(args, "neighbor_radius", 5.0))
    if getattr(args, "actor_neighbor_n", None) is None:
        args.actor_neighbor_n = int(getattr(args, "neighbor_n", 10))
    return args


def compute_undetermined_robot_obs_dim(num_agents: int) -> int:
    """
    Undetermined goal: base 7 + per-goal (dx,dy,in_r,claimed_obs) * K + (K-1) others' tid + pending flag.
    Env appends px, py => full row length is return + 2.
    K == num_agents (swarm size).
    """
    k = int(num_agents)
    return 7 + 4 * k + max(0, k - 1) + 1


def compute_undetermined_v2_robot_obs_dim(goal_slots: int) -> int:
    """
    Undetermined v2: base 7 + M * (dx, dy, in_r, cobs, goal_k_norm) + pending. Px,py appended in env (+2).
    M is fixed by --undetermined_v2_goal_slots (not tied to num_agents in the obs layout).
    """
    m = max(1, int(goal_slots))
    return 7 + 5 * m + 1


def compute_attn_comm_tail_dim(ally_slots: int, human_slots: int, message_dim: int) -> int:
    """Ally geometry + recv messages + humans + obstacle block (no self-7, no px,py)."""
    p = max(0, int(ally_slots))
    h = max(0, int(human_slots))
    m = max(0, int(message_dim))
    return p * 6 + p * m + h * 5 + 4


def compute_undetermined_v2_attn_hybrid_robot_obs_dim(
    goal_slots: int, ally_slots: int, human_slots: int, message_dim: int
) -> int:
    """
    Undetermined v2 core (7 + 5*M + 1) + AttnComm tail; env still appends px,py (+2) to the row.
    """
    return compute_undetermined_v2_robot_obs_dim(goal_slots) + compute_attn_comm_tail_dim(
        ally_slots, human_slots, message_dim
    )


def compute_undetermined_v3_robot_obs_dim(
    goal_slots: int, ally_slots: int, human_slots: int, message_dim: int
) -> int:
    """
    Undetermined v3: v2 nearest-M core + one scalar (prev applied target id norm) before px,py in the core block,
    then the same AttnComm tail; ally/human slot counts are v3 hyperparameters (not tied to num_agents).
    """
    return compute_undetermined_v2_robot_obs_dim(goal_slots) + 1 + compute_attn_comm_tail_dim(
        ally_slots, human_slots, message_dim
    )


def apply_architecture_mode_preset(args) -> None:
    """Reserved for named stacks; flags stay explicit on the CLI (no silent cross-mode coupling)."""
    return


def apply_undetermined_reward_floors(args) -> None:
    """
    When enable_undetermined_goal: emphasize reaching each agent's (gx, gy) over Laplacian formation (S-shape graph),
    cap Hungarian assignment bonus, and keep navigation shaping bounded (clip + distance penalties + timeout).
    """
    if not getattr(args, "enable_undetermined_goal", False):
        return
    args.nd_discount_formation = 0.0
    _nav = float(getattr(args, "nd_discount_nav", 22.0))
    args.nd_discount_nav = min(max(_nav, 22.0), 34.0)
    _gpc = float(getattr(args, "nd_goal_progress_coef", 6.0))
    args.nd_goal_progress_coef = min(max(_gpc, 6.5), 10.5)
    _dg = float(getattr(args, "nd_discount_goal", 200.0))
    args.nd_discount_goal = max(_dg, 320.0)
    _px = float(getattr(args, "nd_proximity_reward_scale", 0.0))
    if _px > 1e-9:
        args.nd_proximity_reward_scale = max(_px, 0.32)
    _hs = float(getattr(args, "nd_heading_reward_scale", 0.0))
    if _hs > 1e-9:
        args.nd_heading_reward_scale = max(_hs, 0.22)
    _tr = float(getattr(args, "nd_goal_terminal_reward", 8.0))
    args.nd_goal_terminal_reward = max(_tr, 22.0)
    _ar = float(getattr(args, "nd_arrival_reward", 4.0))
    args.nd_arrival_reward = max(_ar, 9.0)
    _stay = float(getattr(args, "nd_goal_stay_reward", 0.0))
    args.nd_goal_stay_reward = max(_stay, 4.5)
    _leave = float(getattr(args, "nd_goal_leave_penalty", 14.0))
    args.nd_goal_leave_penalty = max(_leave, 36.0)
    _uh = float(getattr(args, "undetermined_hungarian_reward_scale", 0.15))
    args.undetermined_hungarian_reward_scale = min(_uh, 0.08)
    bst = float(getattr(args, "undetermined_far_goal_progress_boost", 1.6))
    args.undetermined_far_goal_progress_boost = min(bst, 1.14)
    _pdp = float(getattr(args, "undetermined_goal_distance_penalty_scale", 0.0))
    args.undetermined_goal_distance_penalty_scale = max(_pdp, 0.012)
    _pdq = float(getattr(args, "undetermined_goal_dist_penalty_quad_scale", 0.0))
    args.undetermined_goal_dist_penalty_quad_scale = max(_pdq, 0.0001)
    _clip = float(getattr(args, "nd_nav_progress_clip", 0.0))
    if _clip < 1e-9:
        args.nd_nav_progress_clip = 0.42
    else:
        args.nd_nav_progress_clip = min(_clip, 0.55)
    _to = float(getattr(args, "nd_timeout_no_goal_penalty", 0.0))
    if _to > -1e-9:
        args.nd_timeout_no_goal_penalty = -7.0
    else:
        args.nd_timeout_no_goal_penalty = min(_to, -3.0)


def apply_undetermined_v2_reward_floors(args) -> None:
    """
    Undetermined v2 reward profile (run49-style failure modes: c_avoid / heavy dist penalties drowning nav).
    Start from v1 floors then relax avoid coupling, soften world-frame distance pull, strengthen proximity/terminal.
    """
    if not getattr(args, "enable_undetermined_goal", False):
        return
    apply_undetermined_reward_floors(args)
    args.nd_discount_formation = 0.0
    da = float(getattr(args, "nd_discount_avoid", 50.0)) * float(getattr(args, "undetermined_v2_discount_avoid_mult", 0.42))
    args.nd_discount_avoid = max(12.0, min(da, 28.0))
    dn = float(getattr(args, "nd_discount_nav", 22.0)) * 0.88
    args.nd_discount_nav = max(18.0, min(dn, 30.0))
    args.nd_discount_goal = max(float(getattr(args, "nd_discount_goal", 200.0)), 360.0)
    args.nd_goal_progress_coef = max(float(getattr(args, "nd_goal_progress_coef", 6.0)), 7.5)
    px = float(getattr(args, "nd_proximity_reward_scale", 0.0))
    if px > 1e-9:
        args.nd_proximity_reward_scale = max(px, 0.42)
    args.nd_goal_terminal_reward = max(float(getattr(args, "nd_goal_terminal_reward", 8.0)), 26.0)
    args.nd_arrival_reward = max(float(getattr(args, "nd_arrival_reward", 4.0)), 10.0)
    args.nd_goal_stay_reward = max(float(getattr(args, "nd_goal_stay_reward", 0.0)), 5.0)
    args.nd_goal_leave_penalty = max(float(getattr(args, "nd_goal_leave_penalty", 14.0)), 38.0)
    pdp = float(getattr(args, "undetermined_goal_distance_penalty_scale", 0.012))
    args.undetermined_goal_distance_penalty_scale = pdp * float(getattr(args, "undetermined_v2_dist_penalty_mult", 0.38))
    pdq = float(getattr(args, "undetermined_goal_dist_penalty_quad_scale", 0.00012))
    args.undetermined_goal_dist_penalty_quad_scale = pdq * float(getattr(args, "undetermined_v2_dist_penalty_mult", 0.38))
    args.undetermined_far_goal_progress_boost = min(float(getattr(args, "undetermined_far_goal_progress_boost", 1.12)), 1.11)
    args.undetermined_hungarian_reward_scale = min(float(getattr(args, "undetermined_hungarian_reward_scale", 0.08)), 0.07)

    if getattr(args, "undetermined_v2_type2_formation_efficiency", False):
        # Fast formation + short path before type-2 success; after success, allow drift vs literal targets.
        if float(getattr(args, "undetermined_v2_sl_post_success_literal_scale", 1.0)) > 0.999:
            args.undetermined_v2_sl_post_success_literal_scale = 0.28
        if float(getattr(args, "undetermined_v2_sl_post_success_sl_shaping_scale", 1.0)) > 0.999:
            args.undetermined_v2_sl_post_success_sl_shaping_scale = 0.42
        args.undetermined_v2_sl_success_only_on_crossing = True
        if float(getattr(args, "undetermined_v2_sl_success_sustain_frac", 0.0)) < 1e-12:
            args.undetermined_v2_sl_success_sustain_frac = 0.12
        if float(getattr(args, "undetermined_v2_sl_pre_success_step_penalty", 0.0)) > -1e-12:
            args.undetermined_v2_sl_pre_success_step_penalty = -0.018
        if float(getattr(args, "undetermined_v2_sl_pre_success_travel_penalty", 0.0)) > -1e-12:
            args.undetermined_v2_sl_pre_success_travel_penalty = -0.055

    if getattr(args, "undetermined_v2_type2_pattern_first", False):
        # Type-2 / formation emphasis: targets mainly shape L_des; soften per-agent pull to literal (gx, gy).
        args.undetermined_v2_approach_reward_scale = float(
            getattr(args, "undetermined_v2_approach_reward_scale", 0.85)
        ) * 0.48
        args.nd_goal_progress_coef = float(getattr(args, "nd_goal_progress_coef", 7.5)) * 0.82
        args.undetermined_goal_distance_penalty_scale = float(
            getattr(args, "undetermined_goal_distance_penalty_scale", 0.012)
        ) * 0.52
        args.undetermined_goal_dist_penalty_quad_scale = float(
            getattr(args, "undetermined_goal_dist_penalty_quad_scale", 0.00012)
        ) * 0.52
        args.undetermined_far_goal_progress_boost = min(
            float(getattr(args, "undetermined_far_goal_progress_boost", 1.11)), 1.06
        )
        d0 = float(getattr(args, "undetermined_v2_sl_dense_scale", 0.28))
        args.undetermined_v2_sl_dense_scale = max(d0, 0.36)
        s0 = float(getattr(args, "undetermined_v2_sl_success_scale", 2.5))
        args.undetermined_v2_sl_success_scale = max(s0, 3.2)
        if float(getattr(args, "undetermined_v2_sl_delta_scale", 0.0)) < 1e-9:
            args.undetermined_v2_sl_delta_scale = 3.5
        # Light post–type-2-success drift vs literal (gx,gy); full crossing/step/travel preset is
        # --undetermined_v2_type2_formation_efficiency (keeps older pattern_first runs comparable).
        if not getattr(args, "undetermined_v2_type2_formation_efficiency", False):
            if float(getattr(args, "undetermined_v2_sl_post_success_literal_scale", 1.0)) > 0.999:
                args.undetermined_v2_sl_post_success_literal_scale = 0.28
            if float(getattr(args, "undetermined_v2_sl_post_success_sl_shaping_scale", 1.0)) > 0.999:
                args.undetermined_v2_sl_post_success_sl_shaping_scale = 0.42


def compute_dynamic_robot_obs_dim(
    num_agents: int,
    slot_count: int,
    *,
    use_neighbor_attn_lstm_actor: bool = False,
    actor_neighbor_n: int = 10,
) -> int:
    """
    Dynamic goals, fixed-M nearest-target slots (body-frame rel + world goal + masks).
    Per slot: in_radius, k_norm, rel_bx, rel_by, gx, gy, claimed -> 7 * M.
    Either: (K-1) other agents' normalized target_id scalars, or P*7 nearest-neighbor rows
    when use_neighbor_attn_lstm_actor is True.
    """
    k = int(num_agents)
    m = min(max(int(slot_count), 1), k)
    base = 7 + 7 * m
    if use_neighbor_attn_lstm_actor:
        p = max(0, min(int(actor_neighbor_n), max(0, k - 1)))
        return base + p * 7
    return base + max(0, k - 1)


def compute_dynamic_robot_obs_dim_legacy_full_k(num_agents: int) -> int:
    """Older layout: all K goals' rel (2K) + K claim bits + (K-1) others' target_id."""
    k = int(num_agents)
    return 7 + 3 * k + max(0, k - 1)


def compute_attn_comm_robot_obs_dim(
    ally_slots: int, human_slots: int, *, message_dim: int = 0
) -> int:
    """
    Fixed-target layout for AttnCommActorEncoder: self(7) + allies(P*6) + per-slot last-step
    neighbor messages (P*M) + humans(H*5) + obstacle(4). Env appends px, py so full row
    length is return_value + 2.
    """
    p = max(0, int(ally_slots))
    h = max(0, int(human_slots))
    m = max(0, int(message_dim))
    return 7 + p * 6 + p * m + h * 5 + 4


def infer_dynamic_pack_from_actor_feat_dim(
    feat_dim: int,
    num_agents: int,
    preferred_slot_m: Optional[int] = None,
):
    """
    Match actor MLP input width (robot_obs + px,py) to dynamic observation packing.

    Returns (pack_kind, slot_m, neighbor_p_or_none, robot_obs_dim) where pack_kind is
    'legacy' | 'slots' | 'slots_attn', neighbor_p_or_none is P for slots_attn else None,
    or (None, None, None, None) if no known layout fits.
    """
    k = int(num_agents)
    d = int(feat_dim)
    rod_legacy = compute_dynamic_robot_obs_dim_legacy_full_k(k)
    if rod_legacy + 2 == d:
        return "legacy", None, None, rod_legacy

    rem = d - 8 - k
    if rem >= 7 and rem % 7 == 0:
        m = rem // 7
        if 1 <= m <= k:
            rod = 7 + 7 * m + max(0, k - 1)
            if rod + 2 == d:
                return "slots", m, None, rod

    if (d - 9) % 7 != 0:
        return None, None, None, None

    def _try_slots_attn(m_try: int):
        need = d - 9 - 7 * m_try
        if need < 0 or need % 7 != 0:
            return None
        p_try = need // 7
        if not (0 <= p_try <= k - 1):
            return None
        rod = 7 + 7 * m_try + 7 * p_try
        if rod + 2 != d:
            return None
        return m_try, p_try, rod

    if preferred_slot_m is not None:
        m0 = min(max(int(preferred_slot_m), 1), k)
        got = _try_slots_attn(m0)
        if got is not None:
            m_try, p_try, rod = got
            return "slots_attn", m_try, p_try, rod

    for m_try in range(k, 0, -1):
        got = _try_slots_attn(m_try)
        if got is not None:
            m_try, p_try, rod = got
            return "slots_attn", m_try, p_try, rod

    return None, None, None, None
