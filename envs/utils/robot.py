from envs.utils.agent import Agent
from envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, args):
        super().__init__()
        self.collision = False
        self.discomfort_dist = args.dcf_dist
        self.dmin = float('inf')  #agent 与别的agent的最近距离
        self.id = 0
        self.radius = args.robot_radius
        self.L_des = None
        self.edge = args.for_edge
        self.reward_list = []

        self.vx_formation = None
        self.vy_formation = None
        self.for_std = None
        self.pre_theta = None
        self.goal_flag = None

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action, action_indice = self.policy.predict(state)

        return action, action_indice
