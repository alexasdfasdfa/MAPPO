from envs.utils.agent import Agent    
from envs.utils.state import JointState
from policy.policy_human.policy_factory import policy_factory


class Human(Agent):
    def __init__(self, args):
        super().__init__()
        self.policy = policy_factory['orca']()
        self.radius = args.human_radius
        self.dist2rob = None
        self.v_pref = args.v_pref


    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        # print('state',self.get_full_state())
        self.policy.safety_space = 0.1
        action = self.policy.predict(state)
        return action
