import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore
from config.config import (
    compute_attn_comm_robot_obs_dim,
    compute_dynamic_robot_obs_dim,
    compute_dynamic_robot_obs_dim_legacy_full_k,
    compute_undetermined_robot_obs_dim,
    compute_undetermined_v2_robot_obs_dim,
    compute_undetermined_v2_attn_hybrid_robot_obs_dim,
    compute_undetermined_v3_robot_obs_dim,
    resolve_attn_comm_args,
)
import configparser
from .multi_discrete import MultiDiscrete
from envs.utils.utils import reach_goal


class DiscreteActionEnv(gym.Env):
    """
    对于离散动作环境的封装
    Wrapper for discrete action environment.
    """

    def __init__(self, all_args):
        if getattr(all_args, "use_attn_comm_actor", False):
            if getattr(all_args, "enable_dynamic_goal_assignment", False):
                raise ValueError(
                    "use_attn_comm_actor requires fixed (pre-assigned) targets; "
                    "disable --enable_dynamic_goal_assignment."
                )
            resolve_attn_comm_args(all_args)
            _hybrid_ud = (
                getattr(all_args, "enable_undetermined_goal", False)
                and str(getattr(all_args, "architecture_mode", "default")) == "attn_undetermined_goal"
                and (
                    getattr(all_args, "enable_undetermined_goal_v2", False)
                    or getattr(all_args, "enable_undetermined_goal_v3", False)
                )
            )
            if not _hybrid_ud:
                exp = compute_attn_comm_robot_obs_dim(
                    int(all_args.attn_comm_ally_slots),
                    int(all_args.attn_comm_human_slots),
                    message_dim=int(getattr(all_args, "attn_comm_message_dim", 16)),
                )
                if int(getattr(all_args, "robot_obs_dim", 7)) != exp:
                    all_args.robot_obs_dim = exp
        if getattr(all_args, "enable_undetermined_goal", False):
            k = int(all_args.num_agents)
            if getattr(all_args, "enable_undetermined_goal_v3", False):
                m = max(1, int(getattr(all_args, "undetermined_v2_goal_slots", 10)))
                resolve_attn_comm_args(all_args)
                exp = compute_undetermined_v3_robot_obs_dim(
                    m,
                    int(all_args.attn_comm_ally_slots),
                    int(all_args.attn_comm_human_slots),
                    int(getattr(all_args, "attn_comm_message_dim", 16)),
                )
            elif getattr(all_args, "enable_undetermined_goal_v2", False):
                m = max(1, int(getattr(all_args, "undetermined_v2_goal_slots", 10)))
                if (
                    str(getattr(all_args, "architecture_mode", "default")) == "attn_undetermined_goal"
                    and getattr(all_args, "use_attn_comm_actor", False)
                ):
                    resolve_attn_comm_args(all_args)
                    exp = compute_undetermined_v2_attn_hybrid_robot_obs_dim(
                        m,
                        int(all_args.attn_comm_ally_slots),
                        int(all_args.attn_comm_human_slots),
                        int(getattr(all_args, "attn_comm_message_dim", 16)),
                    )
                else:
                    exp = compute_undetermined_v2_robot_obs_dim(m)
            else:
                exp = compute_undetermined_robot_obs_dim(k)
            if int(getattr(all_args, "robot_obs_dim", 7)) != exp:
                all_args.robot_obs_dim = exp
        elif getattr(all_args, "enable_dynamic_goal_assignment", False):
            k = int(all_args.num_agents)
            ver = getattr(all_args, "dynamic_obs_pack_version", "slots")
            if ver == "legacy":
                expected = compute_dynamic_robot_obs_dim_legacy_full_k(k)
            else:
                expected = compute_dynamic_robot_obs_dim(
                    k,
                    int(getattr(all_args, "dynamic_target_slot_count", 1)),
                    use_neighbor_attn_lstm_actor=bool(
                        getattr(all_args, "use_neighbor_attn_lstm_actor", False)
                    ),
                    actor_neighbor_n=int(getattr(all_args, "actor_neighbor_n", 10)),
                )
            if int(getattr(all_args, "robot_obs_dim", 7)) < expected:
                all_args.robot_obs_dim = expected
        self.env = EnvCore(all_args)
        self.random_act_prob = all_args.random_act_prob

        self.robot_num = self.env.robot_num
        self.human_num = self.env.human_num
        self.att_agents = self.env.att_agents
        self.obs_dim = self.env.obs_dim
        self.dir_action_dim = self.env.dir_action_dim
        self.vel_action_dim = self.env.vel_action_dim
        self.time_step = self.env.time_step

        self.is_reset = self.env.is_reset
        self.total_obs = self.env.total_obs
        self.attention_weights = self.env.attention_weights
        self.method = all_args.method
        self.base_v = all_args.base_v

        # configure spaces
        self.action_space = []
        # self.observation_space = []   #gym里面有这个定义
        self.robot_observation_space = []
        self.human_observation_space = []
        self.share_observation_space = []
        # Match SharedReplayBuffer / EnvCore joint layout: num_agents * num_rows * (obs_dim+2)
        _num_rows_cent = 1 + max(self.human_num, self.att_agents)
        share_obs_dim = int(self.robot_num) * _num_rows_cent * (self.obs_dim + 2)

        for agent_idx in range(self.robot_num):
            total_action_space = []

            for_action_space = spaces.Discrete(self.dir_action_dim)
            total_action_space.append(for_action_space)

            vel_action_space = spaces.Discrete(self.vel_action_dim)
            total_action_space.append(vel_action_space)
            if self.env.dynamic_goal_assignment:
                total_action_space.append(spaces.Discrete(self.env.num_goal_targets))

            # total action space 
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            self.robot_observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.obs_dim+2,),
                    dtype=np.float32,
                ))  # [-inf,inf]
        
        for human in range(max(self.human_num, self.att_agents)):
            self.human_observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.obs_dim+2,),  #有,表示长度为self.obs_dim的元组，没有,则就是一个整数
                    dtype=np.float32,
                ))  # [-inf,inf]
            
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.robot_num + max(self.human_num, self.att_agents))
        ]

    def step(self, actions):
        if self.method == 'ppo':
            for i,robot in enumerate(self.env.robots):
                self._set_action(actions[i], robot)

            results = self.env.step()
            obs, rews, dones, infos = results
            return np.stack(obs), np.stack(rews), np.stack(dones), infos

        elif self.method == 'orca':
            for i,robot in enumerate(self.env.robots):
                robot.vx = actions[i][0]
                robot.vy = actions[i][1]
                if reach_goal(robot):
                    robot.vx = 0
                    robot.vy = 0
                    # print('reach_goal')
                    # print(robot.id)
                    if robot.success == None:
                        robot.success = True
                robot.px += robot.vx * self.time_step
                robot.py += robot.vy * self.time_step
            return self.env.step()

        elif self.method == 'apf':
            for i,robot in enumerate(self.env.robots):
                robot.vx = actions[i][0]
                robot.vy = actions[i][1]
                if reach_goal(robot):
                    robot.vx = 0
                    robot.vy = 0
                    # print('reach_goal')
                    # print(robot.id)
                    if robot.success == None:
                        robot.success = True
                robot.px += robot.vx * self.time_step
                robot.py += robot.vy * self.time_step
            return self.env.step()

    def set_comm_broadcasts(self, msgs):
        self.env.set_comm_broadcasts(msgs)

    def apply_undetermined_targets(self, targets):
        self.env.apply_undetermined_targets(targets)
        if getattr(self.env, "undetermined_goal_v3", False):
            K = max(1, int(self.env.num_goal_targets))
            den = float(max(1, K - 1))
            for r in self.env.robots:
                r.undet_prev_tid_norm = float(int(r.target_id) % K) / den
        return self.env.refresh_observations_after_target_change()

    def reset(self):
        obs = self.env.reset()
        return obs

    def seed(self, seed):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    #set each robot actions and update positions
    def _set_action(self, action, robot):
        robot.pre_theta = robot.theta
        dyn = self.env.dynamic_goal_assignment
        if dyn and len(action) > 2:
            k = int(self.env.num_goal_targets)
            nt = int(np.clip(int(action[2]), 0, k - 1))
            old = int(getattr(robot, "target_id", 0))
            robot.target_switched_this_step = nt != old
            robot.prev_target_id = old
            robot.target_id = nt
            if robot.target_switched_this_step:
                prior_sw = int(getattr(robot, "dynamic_episode_target_switch_count", 0))
                robot.dynamic_episode_target_switch_prior = prior_sw
                robot.dynamic_episode_target_switch_count = prior_sw + 1
                robot.dynamic_hold_at_switch = int(getattr(robot, "dynamic_hold_target_steps", 0))
                robot.dynamic_hold_target_steps = 0
            else:
                robot.dynamic_hold_target_steps = int(getattr(robot, "dynamic_hold_target_steps", 0)) + 1

        for i in range(self.dir_action_dim):
            if int(action[0]) == i:
                robot.theta = 2 * np.pi / self.dir_action_dim * i
        for i in range(self.vel_action_dim):
            if int(action[1]) == i:
                robot.v = self.base_v * (i + 1)
            if not dyn and reach_goal(robot):
                robot.v = 0
                if robot.success is None:
                    robot.success = True

        if dyn and (robot.success == True or robot.collision == True):
            robot.v = 0
            return

        robot.px = robot.px + self.time_step * robot.v * np.cos(robot.theta)
        robot.py = robot.py + self.time_step * robot.v * np.sin(robot.theta)

    def set_exchange_network(self, network, device=None):
        """Inject a trained ExchangeNetwork into the underlying env_core."""
        self.env.exchange_network = network
        if device is not None:
            self.env.exchange_device = device
        network.to(device if device else self.env.exchange_device)
        network.eval()

    def close(self):
        """Flush any buffered exchange data before shutting down."""
        collector = getattr(self.env, "exchange_data_collector", None)
        if collector is not None and collector.sample_count > 0:
            collector.flush_to_file(9999999)

    def render(self, mode='vedio', visualize = False):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        from matplotlib import patches
        import matplotlib.lines as mlines

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'  #需要用到系统自带的ffmpeg来生成视频（视频解析工具）

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)  # color library
        goal_robot1_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=4)
        states = self.env.total_obs

        episode_success = False

        if mode == 'human':
            pass

        elif mode == 'vedio':
            if visualize == True:
                fig,ax = plt.subplots(figsize=(7,7))     #设置图像弹窗的长和宽
                ax.tick_params(labelsize=16)  # the params of axis
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 12)
                ax.set_xlabel('x', fontsize=16)
                ax.set_ylabel('y', fontsize=16)

                #这里的设置是初始化，下面会更新
                ## add robot and its goal
                robot_positions = [[state[0][j] for j in range(self.robot_num)] for state in states]
                robots = [plt.Circle(robot_positions[0][i][:2], self.env.robots[i].radius, fill=True)
                        for i in range(self.robot_num)]
                robot_numbers = [plt.text(robots[i].center[0] - x_offset, robots[i].center[1] - y_offset, str(i),
                                color='black', fontsize=12) 
                                for i in range(self.robot_num)]
                robot_labels = [f'Robot {i}' for i in range(len(robots))]
                for i,robot in enumerate(robots):
                    ax.add_artist(robot)
                    ax.add_artist(robot_numbers[i])
                plt.legend(robots, robot_labels, fontsize=16)

                robot_goals = [mlines.Line2D([self.env.robots[i].gx], [self.env.robots[i].gy], 
                                            color=goal_robot1_color, marker='*', linestyle='None',markersize=15, label='Goal_robot1')
                                            for i in range(self.robot_num)]
                goal_labels = [f'RobotGoal {i}' for i in range(len(robot_goals))]
                for i,robot_goal in enumerate(robot_goals):
                    ax.add_artist(robot_goal)
                plt.legend(robot_goals, goal_labels, fontsize=16)

                # add humans and their numbers
                human_positions = [[state[1][j] for j in range(self.human_num)] for state in states]
                humans = [plt.Circle(human_positions[0][i][:2], self.env.humans[i].radius, fill=False)
                        for i in range(self.human_num)]
                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                        color='black', fontsize=12) 
                                        for i in range(self.human_num)]
                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    ax.add_artist(human_numbers[i])

                # add time annotation#注释
                time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)   #在图中显示时间
                ax.add_artist(time)

                # compute attention scores
                if self.attention_weights is not None:
                    attention_scores = [
                        plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                                fontsize=16) for i in range(self.human_num)]

                # compute orientation in each step and use arrow to show the direction
                radius_robot = self.env.robots[0].radius

                #  orientation
                if self.env.robots[0].kinematics == 'unicycle':
                    orientation = [((state[0].px, state[0].py), (state[0].px + radius_robot * np.cos(state[0].theta),
                                    state[0].py + radius_robot * np.sin(state[0].theta))) for state in self.states]
                    orientations = [orientation]
                else:
                    orientations = []
                    for i in range(self.human_num + self.robot_num):
                        orientation = []
                        for state in states:
                            if i < self.robot_num:
                                agent_state = state[0][i]   #robot_state
                            else:
                                agent_state = state[1][i - self.robot_num]
                            theta = np.arctan2(agent_state[0], agent_state[1])
                            ##  different agent
                            orientation.append(((agent_state[0], agent_state[1]), (agent_state[0] + radius_robot * np.cos(theta),
                                                                                agent_state[1] + radius_robot * np.sin(theta))))
                        orientations.append(orientation)

                arrows= [patches.FancyArrowPatch(*orientation[0],color=arrow_color, arrowstyle=arrow_style)
                                for orientation in orientations]
                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0

                def update(frame_num):
                    nonlocal global_step
                    nonlocal arrows
                    global_step = frame_num
                    for i,robot in enumerate(robots):
                        robot.center = robot_positions[frame_num][i]
                        robot_numbers[i].set_position((robot.center[0] - x_offset, robot.center[1] - y_offset))
                    for i, human in enumerate(humans):
                        human.center = human_positions[frame_num][i]
                        human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))#set_position是plt的方法
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                        arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                    time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                anim = animation.FuncAnimation(fig, update, frames=len(states), interval=self.time_step * 1000)
                anim.running = True
                plt.show()

            #calculate success rate
            success_robots = 0
            for robot in self.env.robots:
                if robot.success == True:
                    success_robots += 1
            if success_robots == self.robot_num:
                episode_success = True
        return episode_success
