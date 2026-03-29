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
        default=True,
        help="Robots choose a discrete target index each step; claims + path-length / conflict shaping rewards.",
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
        default=2.0,
        help="Scale for same-target proximity penalty (per pair, split across agents).",
    )
    parser.add_argument(
        "--dynamic_target_switch_penalty",
        type=float,
        default=0.45,
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
        default=0.15,
        help="Shared shaping: reward drop in sum of distances-to-assigned-goals (÷n per agent). Encourages coordinated assignment.",
    )
    parser.add_argument(
        "--dynamic_target_overcommit_scale",
        type=float,
        default=0.4,
        help="Shared penalty for extra agents (beyond one) choosing the same target before success (÷n). Mild negotiation signal.",
    )
    parser.add_argument(
        "--dynamic_arrival_reward",
        type=float,
        default=25.0,
        help="Per-agent bonus the step a robot successfully claims its goal (ensures reaching is reinforced).",
    )
    parser.add_argument(
        "--dynamic_proximity_reward_scale",
        type=float,
        default=0.12,
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
        default=0.2,
        help="Multiply proximity shaping when target switched this step (reduces reward churn near goals). Use 0 to zero.",
    )
    parser.add_argument(
        "--dynamic_switch_near_goal_extra",
        type=float,
        default=0.55,
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
        default=0.18,
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
        "--dynamic_sparse_urgency_max_mult",
        type=float,
        default=1.24,
        help="Max multiplier on r_nav and r_prox when locally sparse (few neighbors within local_density_radius).",
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
        "--dynamic_reciprocal_swap_reward_scale",
        type=float,
        default=0.55,
        help="Per-agent bonus when this step it swaps targets with another agent (each takes the other's previous "
        "target_id) while both move toward each other (heading). 0=off.",
    )
    parser.add_argument(
        "--dynamic_reciprocal_swap_switch_penalty_mult",
        type=float,
        default=0.22,
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
        default=2.5,
        help="Extra switch penalty multiplier ~ (1 + scale/(1+hold_steps)) where hold_steps is time on previous target before switch; discourages rapid target oscillation.",
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
        default=6.0,
        help="Scales (pre_dist2goal - dist2goal) before discount_nav (legacy used 5).",
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
        default=4.0,
        help="Extra r_goal when inside goal disk (any speed); still multiplied by discount_goal.",
    )
    parser.add_argument(
        "--nd_goal_terminal_reward",
        type=float,
        default=8.0,
        help="One-shot bonus the first step the agent enters the goal disk (adds to weighted reward). 0 = legacy (reward 0 every step while in goal).",
    )
    parser.add_argument(
        "--nd_discount_formation",
        type=float,
        default=0.0,
        help="Base weight on r_formation_raw in non-dynamic reward (default 0 = unused). "
        "Effective weight = nd_discount_formation * time-varying factor from formation_time_weight_*.",
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
        default=0.4,
        help="Multiplier on formation weight at episode horizon (tau=1). Linearly interpolated in between.",
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
        default=22.0,
        help="Weight on r_nav_raw (distance progress + optional proximity/heading).",
    )
    parser.add_argument(
        "--nd_discount_goal",
        type=float,
        default=200.0,
        help="Weight on r_goal_raw inside goal disk.",
    )

    parser.add_argument(
        "--randomize_robot_initial_positions",
        action="store_true",
        default=True,
        help="Sample robot starts in a box with minimum separation (dynamic and non-dynamic mode).",
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
    parser.add_argument("--model_dir",type=str,default=None,help="by default None. set the path to pretrained model.",)

    # agent parameters
    parser.add_argument("--num_humans", type=int, default=0, help="number of dynamic obstacles")
    parser.add_argument("--num_attention_agents", type=int, default=5, help="number of agents that should be paid attention")
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
