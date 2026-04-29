import argparse


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
    parser.add_argument("--architecture_mode", type=str, default="default")
    parser.add_argument("--train_font_pattern_length", type=int, default=10)
    parser.add_argument("--train_font_pattern_policy", type=str, default="all")
    parser.add_argument("--time_step",type=float,default=0.1,help='The time interval between each step')
    parser.add_argument("--robot_obs_dim",type=int,default=7,help="robot observation dimension")
    parser.add_argument("--human_obs_dim",type=int,default=5,help="human observation dimension")
    parser.add_argument("--vel_action_dim",type=int,default=5,help="dimension of velocity action space")
    parser.add_argument("--dir_action_dim",type=int,default=18,help="dimension of direction action space")

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
        default="all",
        choices=["all", "nearest_n", "nearest_n_radius"],
        help="How to construct agent-centric inputs: all agents, nearest N, or nearest N within a radius.",
    )
    parser.add_argument(
        "--neighbor_n",
        type=int,
        default=5,
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
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate (default: 5e-4)")
    parser.add_argument("--critic_lr",type=float,default=5e-4,help="critic learning rate (default: 5e-4)",)
    parser.add_argument("--opti_eps",type=float,default=1e-5,help="RMSprop optimizer epsilon (default: 1e-5)",)
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15, help="number of ppo-update epochs (default: 15)")
    parser.add_argument("--use_clipped_value_loss",action="store_false",default=True,
                        help="by default, clip loss value. If set, do not clip loss value.",)
    parser.add_argument("--clip_param",type=float,default=0.2,help="ppo clip parameter (default: 0.2)",)
    parser.add_argument("--num_mini_batch",type=int,default=100,help="number of batches for ppo (default: 1)",)
    parser.add_argument("--entropy_coef",type=float,default=0.01,help="entropy term coefficient (default: 0.01)",)
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
        default=10.0,
        help="max norm of gradients (default: 0.5)",
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
    parser.add_argument("--num_humans", type=int, default=6, help="number of dynamic obstacles")
    parser.add_argument("--num_attention_agents", type=int, default=5, help="number of agents that should be paid attention")
    parser.add_argument("--for_edge", type=int,default=2, help='the formation edge lenth')
    parser.add_argument("--robot_radius", type=float,default=0.3, help='the radius of robot')
    parser.add_argument("--human_radius", type=float,default=0.3, help='the radius of human')
    parser.add_argument("--dcf_dist",type=float,default=0.2,help='discomfort distance of robot and human')
    parser.add_argument("--base_v",type=float,default=0.25,help='When use discrete envirnment,the base velosity in action space')
    parser.add_argument("--robot_initial_spawn_mode", type=str, default="random_box")
    parser.add_argument("--randomize_attributes",type=bool,default=False,help='Randomize humans radius and preferred speed')
    parser.add_argument("--human_action",type=str,default='square_crossing',
                        help='human(dynamic obstacle) act trajectory,include square_crossing,circle_crossing,mixed')
    parser.add_argument("--square_width",type=float,default=10,help='if select square_crossing in human_action,the width of the square')
    parser.add_argument("--circle_radius",type=float,default=10,help='if select circle_crossing in human_action,the radius of the circle')

    # orca parameters
    parser.add_argument("--max_speed", type=float, default=1.25)
    parser.add_argument("--time_horizon",type=int,default=5)
    parser.add_argument("--time_horizon_obst",type=int,default=5)
    parser.add_argument("--neighbor_dist",type=int,default=10)
    parser.add_argument("--max_neighbors",type=int,default=10)
    parser.add_argument("--v_pref",type=float,default=1,help='the preferred velocity of agents')

    # Undetermined goal / AttnComm switches used by MAPPO train.py checks.
    parser.add_argument("--enable_dynamic_goal_assignment", action="store_true", default=False)
    parser.add_argument("--enable_undetermined_goal", action="store_true", default=False)
    parser.add_argument("--enable_undetermined_goal_v2", action="store_true", default=False)
    parser.add_argument("--enable_undetermined_goal_v3", action="store_true", default=False)
    parser.add_argument("--use_attn_comm_actor", action="store_true", default=False)
    parser.add_argument("--undetermined_v2_type2_formation_efficiency", action="store_true", default=False)
    parser.add_argument("--undetermined_v2_goal_slots", type=int, default=10)
    parser.add_argument("--undetermined_target_embed_dim", type=int, default=32)
    parser.add_argument("--undet_v2_head_arch", type=str, default="dot_product")
    parser.add_argument(
        "--undet_v3_head_arch",
        type=str,
        default="global_rank_compat",
        choices=["global_rank_compat", "decoupled_rank_compat", "attention"],
        help="v3 undetermined head architecture; choose *_compat to align with corresponding pretrained latent checkpoints.",
    )
    parser.add_argument("--undet_v3_latent_d_h", type=int, default=64)
    parser.add_argument("--undet_v3_latent_p_max_neighbors", type=int, default=10)
    parser.add_argument("--undet_v3_latent_geo_head_hidden", type=int, default=128)
    parser.add_argument("--undet_v3_decoupled_e_key", type=int, default=64)
    parser.add_argument("--undet_v3_decoupled_e_query", type=int, default=64)
    parser.add_argument("--undet_v3_decoupled_geo_query_dim", type=int, default=16)
    parser.add_argument("--undet_v3_decoupled_tau_logits", type=float, default=0.35)
    parser.add_argument("--undetermined_v3_comm_ally_slots", type=int, default=6)
    parser.add_argument("--undetermined_v3_comm_human_slots", type=int, default=4)
    parser.add_argument("--enable_undetermined_v3_exchange", action="store_true", default=False)
    parser.add_argument("--undetermined_v3_exchange_max_neighbors", type=int, default=10)
    parser.add_argument("--undetermined_v3_exchange_min_m_gain", type=float, default=0.0)
    parser.add_argument("--undetermined_v3_exchange_require_m_gain", action="store_true", default=True)
    parser.add_argument(
        "--disable_undetermined_v3_exchange_require_m_gain",
        action="store_false",
        dest="undetermined_v3_exchange_require_m_gain",
    )
    parser.add_argument(
        "--undetermined_v3_global_dedup_enable",
        action="store_true",
        default=True,
        help="Enable v3 duplicate-target global one-shot deduplication on a consistent full-table snapshot.",
    )
    parser.add_argument(
        "--disable_undetermined_v3_global_dedup",
        action="store_false",
        dest="undetermined_v3_global_dedup_enable",
        help="Disable v3 global one-shot deduplication (fallback to legacy duplicate auction).",
    )
    parser.add_argument(
        "--undetermined_v3_global_dedup_allow_unclaimed_only",
        action="store_true",
        default=True,
        help="When global dedup runs, conflict agents can only switch to duplicated targets or currently unclaimed targets.",
    )
    parser.add_argument(
        "--disable_undetermined_v3_global_dedup_allow_unclaimed_only",
        action="store_false",
        dest="undetermined_v3_global_dedup_allow_unclaimed_only",
        help="Allow global dedup to consider all targets, not just duplicated+unclaimed.",
    )
    parser.add_argument(
        "--undet_v3_latent_dataset_num_agents",
        type=int,
        default=10,
        help="Agent count used by the pretrained undet v3 latent dataset/checkpoint; training auto-aligns to this when v3 latent ckpt is enabled.",
    )
    parser.add_argument(
        "--undet_v3_latent_force_dataset_num_agents",
        action="store_true",
        default=False,
        help="Force MAPPO num_agents to undet_v3_latent_dataset_num_agents when using a v3 latent checkpoint.",
    )
    parser.add_argument("--attn_comm_ally_slots", type=int, default=6)
    parser.add_argument("--attn_comm_human_slots", type=int, default=4)
    parser.add_argument("--attn_comm_message_dim", type=int, default=16)
    parser.add_argument("--attn_comm_radius", type=float, default=5.0)
    parser.add_argument("--undetermined_comm_radius", type=float, default=5.0)
    parser.add_argument("--undetermined_obs_goal_radius", type=float, default=5.0)
    parser.add_argument("--undetermined_hungarian_reward_scale", type=float, default=0.10)
    parser.add_argument("--undetermined_goal_distance_penalty_scale", type=float, default=0.004)
    parser.add_argument("--undetermined_far_goal_progress_boost", type=float, default=1.6)
    parser.add_argument("--nd_discount_avoid", type=float, default=50.0)
    parser.add_argument("--nd_discount_nav", type=float, default=20.0)
    parser.add_argument("--nd_discount_goal", type=float, default=200.0)
    parser.add_argument("--nd_goal_progress_coef", type=float, default=5.0)
    parser.add_argument("--nd_goal_terminal_reward", type=float, default=0.0)
    parser.add_argument("--nd_goal_leave_penalty", type=float, default=0.0)


    return parser


def apply_architecture_mode_preset(args) -> None:
    """Apply lightweight architecture-mode defaults before downstream resolution."""
    mode = str(getattr(args, "architecture_mode", "default"))
    if mode == "attn_undetermined_goal":
        setattr(args, "enable_undetermined_goal", True)
        setattr(args, "use_attn_comm_actor", True)


def resolve_dynamic_target_reasoning_args(args) -> None:
    """Keep dynamic-target and undetermined-goal modes mutually exclusive."""
    if bool(getattr(args, "enable_dynamic_goal_assignment", False)):
        setattr(args, "enable_undetermined_goal", False)


def resolve_attn_comm_args(args) -> None:
    """Fill attn-comm defaults when absent."""
    if getattr(args, "attn_comm_radius", None) is None:
        setattr(args, "attn_comm_radius", float(getattr(args, "undetermined_comm_radius", 6.0)))
    setattr(args, "attn_comm_ally_slots", int(getattr(args, "attn_comm_ally_slots", 6)))
    setattr(args, "attn_comm_human_slots", int(getattr(args, "attn_comm_human_slots", 4)))
    setattr(args, "attn_comm_message_dim", int(getattr(args, "attn_comm_message_dim", 16)))


def resolve_attn_comm_ppo_batch_args(args) -> None:
    if not hasattr(args, "attn_comm_max_ppo_samples_per_gpu"):
        setattr(args, "attn_comm_max_ppo_samples_per_gpu", 200000)


def compute_attn_comm_tail_dim(ally_slots: int, human_slots: int, message_dim: int = 16) -> int:
    p = max(int(ally_slots), 0)
    h = max(int(human_slots), 0)
    md = int(message_dim)
    # Must match env_core._attn_comm_tail_vector:
    # ally geometry (6*P) + recv msg (md*P) + human feats (5*H) + obstacle summary (4).
    return p * 6 + p * md + h * 5 + 4


def compute_attn_comm_robot_obs_dim(ally_slots: int, human_slots: int, message_dim: int = 16) -> int:
    # Base 7-d robot observation + communication tail.
    return 7 + compute_attn_comm_tail_dim(ally_slots, human_slots, message_dim)


def compute_dynamic_robot_obs_dim(
    k: int,
    slot_count: int,
    *,
    use_neighbor_attn_lstm_actor: bool,
    actor_neighbor_n: int,
) -> int:
    slots = max(1, min(int(slot_count), int(k)))
    dim = 7 + slots * 4
    if bool(use_neighbor_attn_lstm_actor):
        dim += max(0, int(actor_neighbor_n)) * 2
    return dim


def compute_dynamic_robot_obs_dim_legacy_full_k(k: int) -> int:
    return 7 + max(int(k), 1) * 4


def compute_undetermined_robot_obs_dim(k: int) -> int:
    return 7 + max(int(k), 1) * 3


def compute_undetermined_v2_robot_obs_dim(m_slots: int) -> int:
    # v2 core = base7 + per-slot5 + pending_flag1 (px,py are appended separately at runtime).
    return 7 + max(int(m_slots), 1) * 5 + 1


def compute_undetermined_v2_attn_hybrid_robot_obs_dim(
    m_slots: int, ally_slots: int, human_slots: int, message_dim: int = 16
) -> int:
    return compute_undetermined_v2_robot_obs_dim(m_slots) + compute_attn_comm_tail_dim(
        ally_slots, human_slots, message_dim
    )


def compute_undetermined_v3_robot_obs_dim(
    m_slots: int, ally_slots: int, human_slots: int, message_dim: int = 16
) -> int:
    # v3 = v2 hybrid core + one extra scalar (prev applied target id norm).
    return compute_undetermined_v2_attn_hybrid_robot_obs_dim(m_slots, ally_slots, human_slots, message_dim) + 1


def apply_undetermined_reward_floors(args) -> None:
    setattr(args, "nd_discount_avoid", float(getattr(args, "nd_discount_avoid", 50.0)))
    setattr(args, "nd_discount_nav", float(getattr(args, "nd_discount_nav", 20.0)))
    setattr(args, "nd_discount_goal", float(getattr(args, "nd_discount_goal", 200.0)))
    setattr(args, "nd_goal_progress_coef", float(getattr(args, "nd_goal_progress_coef", 5.0)))
    setattr(args, "nd_goal_terminal_reward", float(getattr(args, "nd_goal_terminal_reward", 0.0)))
    setattr(args, "nd_goal_leave_penalty", float(getattr(args, "nd_goal_leave_penalty", 0.0)))


def apply_undetermined_v2_reward_floors(args) -> None:
    apply_undetermined_reward_floors(args)
    setattr(args, "undetermined_v2_sl_dense_scale", float(getattr(args, "undetermined_v2_sl_dense_scale", 0.0)))
    setattr(args, "undetermined_v2_sl_delta_scale", float(getattr(args, "undetermined_v2_sl_delta_scale", 0.0)))
    setattr(
        args,
        "undetermined_v2_sl_success_scale",
        float(getattr(args, "undetermined_v2_sl_success_scale", 0.0)),
    )
