import logging
import numpy as np
import rvo2
from numpy.linalg import norm
from envs.utils.human import Human
from envs.utils.info import *
from envs.utils.utils import cal_distance,reach_goal,get_weight
from envs.utils.state_lux import JointState
from envs.utils.robot import Robot
import time
from envs.utils.reward_calculator import RewardCalculator

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self, args):
        self.args = args
        self.time_step = args.time_step
        self.time_limit = args.episode_length * self.time_step
        self.method = args.method
        self.humans = None
        self.robots = None
        self.global_time = None
        self.human_times = None
        self.robot_num = args.num_agents
        self.human_num = args.num_humans
        self.att_agents = args.num_attention_agents
        self.robot_obs_dim = args.robot_obs_dim
        self.human_obs_dim = args.human_obs_dim
        self.obs_dim = max(self.robot_obs_dim, self.human_obs_dim)
        self.vel_action_dim = args.vel_action_dim
        self.dir_action_dim = args.dir_action_dim
        self.discomfort_dist = args.dcf_dist
        self.base_v = args.base_v
        # simulation configuration
        self.config = None
        self.randomize_attributes = args.randomize_attributes
        self.train_val_sim = args.human_action
        self.square_width = args.square_width
        self.circle_radius = args.circle_radius
        # for visualization
        self.total_obs = None

        # for formulate
        self.observation_states = None
        self.attention_weights = None
        self.L_des = None
        self.for_edge = args.for_edge

        self.is_reset = None
        self.collision_flag = None

        # centralized reward computation object (default implementation preserves logic)
        self.reward_calculator = RewardCalculator(self)
        
        # print('human number:',self.human_num)

        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info('simulation: {}'.format(self.train_val_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

        # font pattern selection (used to set robot goals/formation templates)
        self.pattern_name, self.s_shape_targets = self._select_font_pattern_targets()

    def _parse_csv_ints(self, s):
        s = (s or "").strip()
        if not s:
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    def _parse_csv_names(self, s):
        s = (s or "").strip()
        if not s:
            return []
        return [x.strip() for x in s.split(",") if x.strip()]

    def _select_font_pattern_targets(self):
        """
        Select a font pattern (formation template) from dataset.FontPatternLoader.

        Training (use_render == False):
          - lengths: single int from args.train_font_pattern_length
          - threads: args.n_rollout_threads
          - each env_rank selects a pattern (can repeat based on args/train_font_pattern_allow_repeat)

        Render (use_render == True):
          - lengths: possibly multiple from args.render_font_pattern_lengths
          - n_rollout_threads is usually 1; selection still uses env_rank mapping
        """
        from dataset.FontPatternLoader import FontPatternLoader

        use_render = getattr(self.args, "use_render", False)
        if use_render:
            lengths = self._parse_csv_ints(getattr(self.args, "render_font_pattern_lengths", "10"))
            policy = getattr(self.args, "render_font_pattern_policy", "all")
            names = self._parse_csv_names(getattr(self.args, "render_font_pattern_names", ""))
            allow_repeat = True  # render usually uses 1 env, repetition doesn't matter much
        else:
            lengths = [int(getattr(self.args, "train_font_pattern_length", 10))]
            policy = getattr(self.args, "train_font_pattern_policy", "all")
            names = self._parse_csv_names(getattr(self.args, "train_font_pattern_names", ""))
            allow_repeat = bool(getattr(self.args, "train_font_pattern_allow_repeat", False))

        env_rank = int(getattr(self.args, "env_rank", 0))
        threads = int(getattr(self.args, "n_rollout_threads", 1))

        loader = FontPatternLoader(target_lengths=lengths, dataset_dir=getattr(self.args, "font_pattern_dataset_dir", "./dataset"))

        pool_all = []
        for L in lengths:
            pool_all.extend(loader.data_store.get(L, []))

        if not pool_all:
            raise RuntimeError(f"No font patterns loaded for lengths={lengths}.")

        # apply policy filter
        if policy == "only":
            if not names:
                raise RuntimeError("train_font_pattern_policy/render_font_pattern_policy is 'only' but no names provided.")
            pool_allowed = [p for p in pool_all if p.get("name") in set(names)]
        else:
            pool_allowed = list(pool_all)

        if not pool_allowed:
            raise RuntimeError(f"Font pattern policy '{policy}' produced empty pool. names={names}, lengths={lengths}")

        # must_contain: reorder so required names come first
        ordered_pool = []
        used_names = set()
        if policy == "must_contain":
            if not names:
                raise RuntimeError("Font pattern policy 'must_contain' requires non-empty pattern names.")

            required_set = set(names)
            found_set = {p.get("name") for p in pool_allowed}
            missing = required_set - found_set
            if missing:
                raise RuntimeError(f"Missing required font patterns: {sorted(list(missing))}")

            if len(names) > threads:
                raise RuntimeError(
                    f"Not enough env threads (threads={threads}) to include all required patterns (required={names})."
                )

            for n in names:
                # pick first matching pattern (names are expected unique in dataset)
                for p in pool_allowed:
                    if p.get("name") == n:
                        ordered_pool.append(p)
                        used_names.add(n)
                        break

        # append the rest
        for p in pool_allowed:
            n = p.get("name")
            if n not in used_names:
                ordered_pool.append(p)

        pattern_count = len(ordered_pool)
        if threads > pattern_count:
            allow_repeat = True  # requirement: can repeat when threads exceed patterns

        if not allow_repeat and threads > pattern_count:
            raise RuntimeError("Repetition disabled but env threads exceed available patterns.")

        if allow_repeat:
            idx = env_rank % pattern_count
        else:
            idx = env_rank

        chosen = ordered_pool[idx]
        pattern_name = chosen.get("name", "unknown")
        pattern_coords = chosen.get("coordinates", [])

        if not pattern_coords:
            raise RuntimeError(f"Chosen font pattern '{pattern_name}' has empty coordinates.")

        if not use_render:
            print(f"[font-pattern][train] env_rank={env_rank}/{threads-1}, length={lengths[0]}, name={pattern_name}")
        else:
            print(f"[font-pattern][render] env_rank={env_rank}, length(s)={lengths}, name={pattern_name}")

        return pattern_name, pattern_coords

    def generate_remote_human_position(self):
        for i,human in enumerate(self.humans):
            human.set(10e5+i, 10e5+i, 10e5+i, 10e5+i, 0, 0, 0)

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.args)
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.args)
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in self.robots + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_square_crossing_human(self):
        human = Human(self.args)
        radius = human.radius
        if self.randomize_attributes:
            human.sample_random_attributes()
        sign = np.random.random()
        revise = 0
        while True:
            px = np.random.uniform(-10,10) + revise
            py = np.random.uniform(-10,10) + revise
            collide = False
            for agent in self.robots + self.humans:
                if norm((px - agent.px, py - agent.py)) < (radius + agent.radius + self.discomfort_dist)*2 or \
                    norm((px - agent.gx, py - agent.gy)) < (radius + agent.radius + self.discomfort_dist)*2:
                    collide = True
                    break
            if not collide:
                break
            revise += 1

        revise = 0
        while True:
            if sign >= 0.7: # 0.7
                gx = np.random.uniform(-10,10) - 1 -revise
                gy = np.random.uniform(-10,10) + 1 + revise
            elif 0.3 <= sign < 0.7:   # 0.3~0.7
                gx = px
                gy = py
            else:
                gx = px - self.square_width/2 - revise
                gy = px - self.square_width/2 - revise
            collide = False
            for agent in self.robots + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
            revise += 1
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_circle_crossing_human(self):
        human = Human(self.args)
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            #  check robot collide
            for agent in self.robots + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human
        
    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.robots = [Robot(self.args) for i in range(self.robot_num)]
        self.total_obs = []
        # print('env has reset!')
        if self.robots is None:
            raise AttributeError('robots has not set!')
        else:
            self.global_time = 0
                
            #initialize robots
            self.collision_flag = False
            set_bias = np.random.random() * 0.5
            px = -5
            py = -5
            coordinates_list = [(4,6),(3,6),(2,6),(1,6),(0,6),(-1,6),(-2,6),(4,5),(3,5),(1,5),(-1,5),(-5,4),(-5,6)]
            # coordinates_list = [(2,2), (2,1.5), (2,1), (2,0.5)]
            random_index = np.random.randint(len(coordinates_list))
            random_coordinate = coordinates_list[random_index]
            gx = random_coordinate[0] + set_bias
            gy = random_coordinate[1] + set_bias
            # 默认起始位置偏移（保留原有起始点/目标增长逻辑）
            n = 0
            # 如果你想使用自定义的 S 型队形目标点（绝对坐标），在这里列出目标坐标。
            # 下面是用户提供的 S 型目标点列表（10 个点）：
            # candidate_targets = [
            #     [[1, 26] ,[1, 22], [ 1, 18],[ 1, 14],[ 1, 10],[ 1,  6], [ 4, 4], [ 7,  3], [ 10, 4], [ 13, 6], [ 13, 10],[ 13, 14],[ 13, 18], [13,22],[13, 26]], #U
            #     [[1, 26], [3, 26], [5, 26], [7, 26], [9, 26], [11, 26], [13, 26], [7, 15], [7, 12], [7, 9], [7, 6], [7, 3], [7, 17], [7, 20], [7, 23]], #T
            #     [[1,26],[1,22],[1,18],[1,11],[1,7],[1,3],[5,26],[9,26],[13,26],[1,15],[5,15],[9,15],[5,3],[9,3],[13,3]], #E
            #     [[13,25],  [10, 26], [7,26], [4,25],[2,23], [1, 19], [1, 15], [1, 11], [1, 7], [3, 4], [6,3], [8, 3],[10, 3], [12, 4], [13, 6]], #C
            #     [[12, 24], [9, 26], [6, 26], [3, 24], [1, 19], [4, 16], [7, 15], [10, 13], [12, 11], [13, 8], [11, 5], [9, 4], [7, 3], [4,4], [1, 5]], #S
            #     [[1, 26],[1, 22], [1,13], [1,8], [1, 3],[1,18],[4,18],[7,18],[10,18],[13,18], [13, 26],[13, 22], [13,13], [13,8], [13,3]], #H
            # ]
            candidate_targets = [
                [[1, 26] , [ 1, 18],[ 1, 12],[ 1,  6], [ 5, 3], [ 9, 3], [ 13, 6], [ 13, 12],[ 13, 18], [13, 26]]  ,      #U
 [[1, 26], [4, 26], [7, 26], [10, 26], [13, 26], [7, 23], [7, 18], [7, 13], [7, 8], [7, 3]]                        ,      #T                                       
 [[1,26],[1,18],[1,11],[1,3],[7,26],[13,26],[5,15],[9,15],[7,3],[13,3]]                                            ,      #E           
 [[12, 25], [7,26], [3,24], [1, 19], [1, 15],  [1, 10], [3, 4], [6,3], [10, 3],  [13, 6]]                          ,      #C               
[[12, 24], [7, 26], [3, 24], [1, 19], [5, 15],  [11, 12],  [13, 8], [10, 4],  [6,4], [1, 5]]                       ,      #S                               
 [[1, 26],[1, 11], [1, 3],[1,19],[5,18],[9,18],[13,19], [13, 26],[13, 11], [13,3]]                                 ,      #H                           
 [[7,26],[7,16],[2,16],[12,16],[4,20], [10, 20],[1,11],[1,3],[13,11],[13,3]]                                       ,      #A                                                          
 [[3, 26], [1, 21], [1, 15], [1,9], [1,3], [8, 26], [7,15],[7,3], [12, 21],[13,9]]                                 ,      #B           
[[3, 26], [1, 21], [1, 15], [1,9], [1,3], [8, 26], [13,15],[7,3], [12, 21],[11,7]]                                 ,      #D                                                                                                         
[[1,26],[1,21],[1,15],[1,9], [1,3],[5,26],[9,26],[13,26],[5,15],[9,15]]                                            ,      #F               
[ [8,26], [3,24], [1, 17],   [1, 10], [5, 3], [9, 3],  [13, 7],[13, 12],[9, 12],[13,3], ]                          ,      #G                               
 [[7,26],[7,21], [7,16],[7,11],[7,7],[7,3],[6, 26],[8,26],[5,3],[9,3]]                                             ,      #I                       
 [ [10,26],[13, 26], [13,22],  [13, 18], [13, 14],  [13, 10], [3, 4], [6,3], [10, 3],  [13, 6],]                   ,      #J                                                   
 [[1,26],[1,20], [1,15],[1,10],[1,3],[10,26], [6, 19],[5, 12],[9, 8],[13,3]]                                       ,      #K                               
 [[1,26],[1,21], [1,16],[1,11],[1,7],[1,3], [4,3],[7,3],[10,3],[13,3]]                                             ,      #L               
 [[1,3],[1,17],[3, 26],[5,26], [7,15],[7,5],[9,26],[11,26],[13,17],[13,3]]                                         ,      #M                                   
[[13,26],[12, 19], [11, 11],[10, 3], [8,11],[6,19],[4,26],[3, 19],[2,11],[1,3]]                                    ,      #N                                                                             
[[5, 26], [2, 21], [1, 15], [2,9], [9,3], [9, 26], [13,15],[5,3], [12, 21],[12,9]]                                 ,      #O               
[[1, 26], [1, 21], [1, 15], [1,9], [1,3],[6,26], [10, 26], [13,22],[10,17],[6,17]]                                 ,      #P                           
[ [3, 20],  [2,11], [9,3], [7, 26], [5,3], [11, 20],[12, 11],[8,8], [11,6] ,[13,3]]                                ,      #Q               
[[1, 26], [1, 18],  [1,10], [1,3],[6,26], [10, 26], [12,21],[6,17],[8,11],[13,3]]                                  ,      #R                                       
[[1, 26] , [ 1, 21],[ 1, 16],[ 4,  11], [ 7, 3], [ 7, 6], [ 10, 11], [ 13, 16],[ 13, 21], [13, 26]]                ,      #V                                                           
[[1,26],[1,12],[3, 3],[5,3], [7,14],[7,24],[9,3],[11,3],[13,12],[13,26]]                                           ,      #W                       
[[1,26],[4, 21],[7,13],[10,9],[13,3],[1,3],[4,9],[7,17],[10,21],[13,26]]                                           ,      #X                   
[[1, 26] , [ 3, 21],[ 5, 16], [ 7, 3], [ 7, 6], [7,9],[7,12], [ 9, 16],[ 11, 21], [13, 26]]                        ,      #Y               
[[1,26], [5,26],[9,26],[13,26],[9,19],[5,11],[1,3],[5,3],[9,3],[13,3]]                                             ,      #Z           
            ]                                                                                                                                               
            # s_shape_targets = candidate_targets[np.random.randint(0, len(candidate_targets))]
            s_shape_targets = self.s_shape_targets
            # s_shape_targets = candidate_targets[ [ 1,4][np.random.randint(0, 2)]]
            # s_shape_targets = candidate_targets[ [0, 1, 2, 4][np.random.randint(0, 2)]]
            # s_shape_targets = candidate_targets[ [0, 1,2,4 ,3,5,6,7][np.random.randint(0, 8)]]
            # s_shape_targets = candidate_targets[list(range(0,16))[np.random.randint(0, 16)]]
            # 选择用于赋值的目标点数目（如果机器人数量少于点，截断；多于点，重复最后一个点）
            used_targets = []
            if self.robot_num <= len(s_shape_targets):
                used_targets = s_shape_targets[: self.robot_num]
            else:
                used_targets = list(s_shape_targets)
                # 若机器人数量超过目标点数，则重复最后一个点作为占位
                while len(used_targets) < self.robot_num:
                    used_targets.append(s_shape_targets[-1])

            # 将目标点转换为相对于质心的相对模板（for_std），以便队形对平移不敏感
            centroid = np.mean(np.array(used_targets), axis=0)
            used_targets = sorted(used_targets, key= lambda target: ((np.arctan(1/np.divide(*(target-centroid))) + (np.pi if (target-centroid)[0]<0 else 0))*2/np.pi+4)%4)
            rel_targets = [(tx - centroid[0], ty - centroid[1]) for (tx, ty) in used_targets]

            for i, robot in enumerate(self.robots):
                robot.vx_formation = 0
                robot.vy_formation = 0
                # 保持原有的 set 起始位置（可按需调整 px,py/gx,gy 的生成策略）
                robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
                px += robot.edge*np.cos(2*np.pi*i/10)#(robot.edge / 2) * np.sqrt(3) * (-1) ** n
                py += robot.edge*np.sin(2*np.pi*i/10)#(robot.edge / 2)
                # py += robot.edge
                robot.gx = gx + rel_targets[i][0]#(robot.edge / 2) * np.sqrt(3) * (-1) ** n
                robot.gy = gy + rel_targets[i][1]#(robot.edge / 2) 
                # gy += robot.edge
                n += 1
                robot.v = 0
                robot.theta = 0
                robot.dmin = float('inf')
                robot.id = 0
                robot.collision = None
                robot.success = None
                # 将 for_std 设为相对于质心的目标位置（用于 formation 误差计算）
                robot.for_std = [rel_targets[i][0], rel_targets[i][1]]
            
            for robot in self.robots:
                for r in self.robots:
                    if robot != r:
                        robot.vx_formation -= (robot.px - r.px - (robot.for_std[0] - r.for_std[0]))
                        robot.vy_formation -= (robot.py - r.py - (robot.for_std[1] - r.for_std[1]))
            
            # W = (np.ones((self.robot_num, self.robot_num)) - np.eye(self.robot_num)) * self.for_edge
            
            # Build adjacency/weight matrix using the actual targets assigned to robots
            # (used_targets length equals number of robots considered). Using s_shape_targets
            # here caused L_des to have a size mismatch when robot_num != len(s_shape_targets).
            W = np.array([[get_weight(*c1, *c2) for c1 in used_targets] for c2 in used_targets])

            # Degree matrix: diagonal of row-sums of W
            row_sums = np.sum(W, axis=1)
            D = np.diag(row_sums)
            L = D - W
            D_sys = np.diag(np.power(row_sums, -1.0 / 2.0))  # symmetric normalize
            self.L_des = D_sys @ L @ D_sys
            assert np.round(self.L_des[0][0]) == 1,'L compute error!'

            #原本是根据不同的case初始化障碍物
            self.generate_random_human_position(human_num=self.human_num, rule=self.train_val_sim)
            self.generate_remote_human_position()

            #设置时间步长
            for agent in self.robots + self.humans :
                agent.time_step = self.time_step
            
            obs = []
            temp_robot_obs = []
            temp_human_obs = []
            for i,robot in enumerate(self.robots):
                temp_obs = np.zeros(((1 + max(self.human_num, self.att_agents)), self.obs_dim+2))
                #TODO formation compute
                temp_obs[0,:self.robot_obs_dim+2] = \
                np.array([robot.gx-robot.px, robot.gy-robot.py, robot.v, robot.theta, 0, robot.vx_formation, robot.vy_formation, robot.px, robot.py]).copy()

                temp_robot_obs.append(
                np.array([robot.px, robot.py, robot.gx, robot.gy, robot.v, robot.theta, 0, robot.vx_formation, robot.vy_formation]).copy())
                
                for human in self.humans:
                    human.dist2rob = cal_distance(robot.px, robot.py, human.px, human.py)
                self.humans = sorted(self.humans, key=lambda x:x.dist2rob, reverse=True)   #行人距离机器人的距离由远到近进行排序
                assert self.humans[-1].dist2rob < self.humans[-2].dist2rob, 'sort error!'

                for j, human in enumerate(self.humans[:self.human_num]):
                    htheta = np.arctan2(human.py, human.px)
                    temp_obs[j + 1,:self.human_obs_dim] = np.array([human.px, human.py, human.vx, human.vy, htheta]).copy()
                obs.append(temp_obs)
                

                for human in self.humans:
                    temp_human_obs.append(np.array([human.px, human.py, human.vx, human.vy, htheta]).copy())

            self.total_obs.append([temp_robot_obs, temp_human_obs])
            
            obs_orca = []
            for agent in self.robots + self.humans:
                obs_orca.append(np.array([agent.px, agent.py, agent.vx, agent.vy, agent.gx, agent.gy]))

        if self.method == 'ppo':
            return obs
        elif self.method == 'orca' or self.method == 'apf':
            return obs_orca


    def step(self):
        sub_agent_obs = []
        sub_agent_obs_render = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        human_obs = []
        human_actions = []
        id_index = 0   #for formation encode
        reachgoal_num = 0   #for calculate the success rate

        #human action
        for human in self.humans:
             # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            #这里得到的ob是JointState中的human_states，即其它human的状态
            human_action = human.act(ob)
            human_actions.append(human_action)    #human到终点后位置不再更新
            human.theta = np.arctan2(human.vy,human.vx)
            human_obs.append(np.array([human.px, human.py, human.vx, human.vy, human.theta]))
        # for i, human_action in enumerate(human_actions):
        #     self.humans[i].step(human_action)
        
        #for reward calculate
        W = np.zeros((self.robot_num, self.robot_num))
        for i, robot in enumerate(self.robots):
            #initialize some flags of robots
            robot.dmin = float('inf')
            robot.collision = False
            robot.vx_formation = 0
            robot.vy_formation = 0

            for j, car in enumerate(self.robots):
                W[i][j] = get_weight(robot.px, robot.py, car.px, car.py)

            # collision  detect
            for agent in self.robots + self.humans:
                d = cal_distance(robot.px, robot.py, agent.px, agent.py)
                if agent != robot and d < robot.dmin:
                    robot.dmin = cal_distance(robot.px, robot.py, agent.px, agent.py)
                    
            # if robot.dmin - robot.radius - agent.radius < 0 or robot.px < 0 or robot.py < 0: # with boundary
            if robot.dmin - robot.radius - agent.radius < 0:   # without boundary
                robot.collision = True
                # print('collision')
                robot.success = False
                self.collision_flag = True
            else:
                robot.dmin = robot.dmin - robot.radius - agent.radius

            #formation detect
            id_index += 1
            robot.id = id_index

            for robot in self.robots:
                for r in self.robots:
                    if robot != r:
                        robot.vx_formation -= (robot.px - r.px - (robot.for_std[0] - r.for_std[0]))
                        robot.vy_formation -= (robot.py - r.py - (robot.for_std[1] - r.for_std[1]))
            # print('robot.px,robot.py',robot.px,robot.py)

        assert W[-1][-2] != 0,'W compute error!'

        # print(W)
        D = np.diag(np.sum(W, axis=1))  # degree matrix via row sums
        L = D - W
        # symmetric normalization: D_sys = diag(row_sums^{-1/2}); guard against zeros
        row_sums = np.sum(W, axis=1)
        eps = 1e-8
        inv_sqrt = np.power(np.where(row_sums > 0, row_sums, eps), -0.5)
        D_sys = np.diag(inv_sqrt)
        L_hat = D_sys @ L @ D_sys  # scaled Laplacian

        # Ensure self.L_des has the same shape as L_hat. If not, rebuild L_des from current robot goals.
        if (self.L_des is None) or (self.L_des.shape != L_hat.shape):
            # Build W_des from current robots' goal positions (gx, gy)
            try:
                targets = [(r.gx, r.gy) for r in self.robots]
                W_des = np.array([[get_weight(tx1, ty1, tx2, ty2) for (tx1, ty1) in targets] for (tx2, ty2) in targets])
                row_sums_des = np.sum(W_des, axis=1)
                inv_sqrt_des = np.power(np.where(row_sums_des > 0, row_sums_des, eps), -0.5)
                D_des = np.diag(row_sums_des)
                L_des = D_des - W_des
                D_sys_des = np.diag(inv_sqrt_des)
                self.L_des = D_sys_des @ L_des @ D_sys_des
            except Exception:
                # If rebuilding fails for any reason, raise a clear error with shapes for debugging
                raise RuntimeError(f"L_des shape mismatch and rebuild failed: L_hat.shape={L_hat.shape}, L_des.shape={(None if self.L_des is None else self.L_des.shape)}")

        for_feature = np.trace(np.transpose((L_hat - self.L_des)) @ (L_hat - self.L_des))
        # print(for_feature)

        #navigation
        for robot in self.robots:
            if cal_distance(robot.px,robot.py,robot.gx,robot.gy) <= robot.radius:
                reachgoal_num += 1
            
            robot.pre_dist2goal = robot.dist2goal
            robot.dist2goal = cal_distance(robot.px,robot.py,robot.gx,robot.gy)

        self.global_time += self.time_step
        reward = 0
        for i, robot in enumerate(self.robots):
            sub_agent_obs.append(self.get_obs(robot, self.humans, for_feature)[0])
            sub_agent_obs_render.append(self.get_obs(robot, self.humans, for_feature)[1])
            sub_agent_reward.append(self.get_reward(robot,for_feature))
            # reward += self.get_reward(robot,for_feature)

            sub_agent_done.append(self.get_done(robot))
            sub_agent_info.append(self.get_info(robot))
        
        # sub_agent_reward = [reward] * len(self.robots)
        sub_agent_obs = np.array(sub_agent_obs)
        #for visualize(all step obs)
        self.total_obs.append([sub_agent_obs_render, human_obs])  #dim(step, 2, agent_num, obs_dim)

        if self.collision_flag == True:
            sub_agent_done = [True] * self.robot_num

        obs_orca = []
        for agent in self.robots + self.humans:
            obs_orca.append(np.array([agent.px, agent.py, agent.vx, agent.vy, agent.gx, agent.gy]))

        if self.method == 'ppo':
            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
        if self.method == 'orca' or self.method == 'apf':
            return  [obs_orca]
    
    def get_obs(self, robot, humans, for_feature):
        obs = np.zeros(((1 + max(self.human_num, self.att_agents)), self.obs_dim+2))

        # px = robot.px + self.time_step * robot.v * np.cos(robot.theta)
        # py = robot.py + self.time_step * robot.v * np.sin(robot.theta)
        px = robot.px
        py = robot.py
        gx = robot.gx - px
        gy = robot.gy - py

        v = robot.v
        theta = robot.theta
        vx_for = robot.vx_formation
        vy_for = robot.vy_formation

        obs[0,:self.robot_obs_dim+2] = np.array([gx, gy, v, theta, for_feature, vx_for, vy_for, px, py])
        obs_render = np.array([px, py, gx, gy, v, theta, for_feature, vx_for, vy_for])

        for human in humans:
            human.dist2rob = cal_distance(robot.px, robot.py, human.px, human.py)
        humans = sorted(humans, key=lambda x:x.dist2rob, reverse=True)   #行人距离机器人的距离由远到近进行排序
        assert humans[-1].dist2rob <= humans[-2].dist2rob, 'sort error!'

        for j,human in enumerate(humans[:self.human_num]):
            obs[j + 1,:self.human_obs_dim] = \
            np.array([human.px, human.py, human.vx, human.vy, np.arctan2(human.py,human.px)]).copy()
        return [obs, obs_render]
    
    def get_reward(self,robot,for_feature):
        # Keep EnvCore.get_reward() signature, but delegate computation.
        return self.reward_calculator.compute_default_reward(robot, for_feature)
    
    def get_done(self,agent):
        done = False
        if agent.collision == True:
            done = True
        
        if reach_goal(agent):
            done = True

        if self.global_time >= self.time_limit:
            done = True

        return done
    
    def get_info(self,agent):
        info = Nothing()
        if agent.collision == True:
            info = Collision()

        if agent.dmin < agent.discomfort_dist:
            info = Danger()

        if self.global_time >= self.time_limit:
            info = Timeout()
        
        if reach_goal(agent):
            info = ReachGoal()

        return info
    
    
