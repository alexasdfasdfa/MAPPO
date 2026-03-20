import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ENVS.envs.utils.info import *
from ENVS.envs.utils.state import FullState


def average(input_list):
    if input_list:
        return sum(np.array(input_list).tolist()) / len(input_list)
    else:
        return 0


class Explorer(object):  # run k epis to update buffer in training; to test the model in eva/testing
    def __init__(self, args, env, robot1, robot2, robot3, device, memorys, gamma, policy, model, batch_size, train_batches, num):
        self.env = env
        self.robot1 = robot1
        self.robot2 = robot2
        self.robot3 = robot3
        self.device = device
        self.gamma = gamma
        self.policy = policy
        self.model = model

        # self.model_critic = critic
        self.args = args
        self.buffer = memorys
        self.num = num

        self.batch_size = batch_size
        self.trainer = Trainer(self.model, self.buffer, self.device, self.batch_size, self.policy, self.num)
        self.train_batches = train_batches

        self.total_steps_robot1 = []
        self.total_steps_robot2 = []
        self.total_steps_robot3 = []

        self.data_loader = None

        self.total_rewards_robot1 = []
        self.total_rewards_robot2 = []
        self.total_rewards_robot3 = []
        self.optimizer_ac = []
        for i in range(num):
            keyname = 'model{}'.format(i)
            self.optimizer_ac.append(optim.Adam(self.model[keyname].parameters(), lr=3e-6))
        self.criterion = nn.MSELoss().to(device)
        self.optimizer = []

    def set_learning_rate(self, learning_rate):
        #logging.info('Current learning rate: %f', learning_rate)
        for i in range(self.num):
            keyname = 'model{}'.format(i)
            self.optimizer.append(optim.SGD(self.model[keyname].parameters(), lr=learning_rate, momentum=0.9))

    # def act(self, ob):
    #     action_robot1, action_robot2, action_robot1_indice, action_robot2_indice\
    #         = self.policy.predict(ob)

    #     return action_robot1, action_robot2, action_robot1_indice, action_robot2_indice

    def compute_returns_robot1(self, rewards, masks):
        R = 0
        returns = []
        #        reversed , 从后往前排序
        for i in reversed(range(len(rewards))):
            gamma_bar = pow(self.gamma, self.robot1.time_step * self.robot1.v_pref)
            R = rewards[i] + gamma_bar * R * masks[i]
            returns.insert(0, R)
        return returns

    def compute_returns_robot2(self, rewards, masks):
        R = 0
        returns = []
        #        reversed , nipaixu
        for i in reversed(range(len(rewards))):
            gamma_bar = pow(self.gamma, self.robot2.time_step * self.robot2.v_pref)
            R = rewards[i] + gamma_bar * R * masks[i]
            returns.insert(0, R)
        return returns



    def update_buffer(self, states, robot1_rewards, robot2_rewards, robot3_rewards, imitation_learning=False):
        if self.buffer is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward_robot1 = robot1_rewards[i]
            reward_robot2 = robot2_rewards[i]
            reward_robot3 = robot3_rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                value1 = sum([pow(self.gamma, max(t - i, 0) * self.robot1.time_step * self.robot1.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(robot1_rewards)])
                value2 = sum([pow(self.gamma, max(t - i, 0) * self.robot2.time_step * self.robot2.v_pref) * reward
                              * (1 if t >= i else 0) for t, reward in enumerate(robot2_rewards)])
                value3 = sum([pow(self.gamma, max(t - i, 0) * self.robot3.time_step * self.robot3.v_pref) * reward
                              * (1 if t >= i else 0) for t, reward in enumerate(robot3_rewards)])
                ''''''
                state1 = self.policy.transform(state, 0)
                value1 = torch.Tensor([value1]).to(self.device)
                self.buffer[0].push((state1, value1))
                state2 = self.policy.transform(state, 1)
                value2 = torch.Tensor([value2]).to(self.device)
                self.buffer[1].push((state2, value2))
                state3 = self.policy.transform(state, 2)
                value3 = torch.Tensor([value3]).to(self.device)
                self.buffer[2].push((state3, value3))
            else:
                if i == len(states) - 1:
                    # terminal state
                    value1 = reward_robot1
                    value2 = reward_robot2
                    value3 = reward_robot3
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot1.time_step * self.robot1.v_pref)
                    state1 = self.policy.transform(next_state, 0).to(self.device,torch.float32)
                    state2 = self.policy.transform(next_state, 1).to(self.device,torch.float32)
                    state3 = self.policy.transform(next_state, 2).to(self.device,torch.float32)                
                    value1 = reward_robot1 + gamma_bar * self.model['model0'](state1.unsqueeze(0)).data.item()
                    value1 = torch.Tensor([value1]).to(self.device,torch.float32) 
                    self.buffer[0].push((state1, value1))
                    value2 = reward_robot2 + gamma_bar * self.model['model1'](state2.unsqueeze(0)).data.item()
                    value2 = torch.Tensor([value2]).to(self.device) 
                    self.buffer[1].push((state2, value2))
                    value3 = reward_robot3 + gamma_bar * self.model['model2'](state3.unsqueeze(0)).data.item()
                    value3 = torch.Tensor([value3]).to(self.device) 
                    self.buffer[2].push((state3, value3))


    # def add_episode_robot1_and_robot2(self, trajectory):
    #     for (state, A1, R1, A2, R2) in trajectory:
    #         # action_indice = np.array(action_indice)
    #         # R = R.squeeze(0).detach().numpy()
    #         self.buffer.add(state, A1, R1, A2, R2)


    def _update_network(self, num_epochs, imitation_learning=False):
        # print('states', states)
        if imitation_learning == True:
            ada_ent = 0.01
        else:
            ada_ent = 0.001
        self.set_learning_rate(ada_ent)
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = []
            for i in range(self.num):
                self.data_loader.append(DataLoader(self.buffer[i], self.batch_size, shuffle=True))

        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(len(self.data_loader)):
                keyname = 'model{}'.format(int(i))
                average_epoch_losses = []
                average_epoch_loss = 0
                for data in self.data_loader[i]:
                    states, robot_returns = data
                    states = torch.tensor(states)

                    self.set_learning_rate(ada_ent)
                    returns_robot = torch.tensor(robot_returns)
                   
                    states = states.to(self.device)
                                                                                    
                    self.optimizer_ac[i].zero_grad()
                    #logging.info('states:%s',states)
                    value = self.model[keyname](states.to(self.device))
                    
                    value_loss = self.criterion(value, returns_robot)
                    value_loss.backward()

                    self.optimizer_ac[i].step()  # 用于执行反向传播和更新模型参数的函数
                    epoch_loss += value_loss.data.item()
                    average_epoch_loss = epoch_loss / len(self.buffer[i])
                    #logging.info('%s',self.model[keyname].state_dict())
                logging.debug('Average total_loss in epoch %d: %.2E', epoch, average_epoch_loss)
                average_epoch_losses.append(average_epoch_loss)
        return average_epoch_losses

        # print('total_loss', total_loss)
        # print('action_loss', action_loss, 'value_loss', value_loss, 'total_loss', total_loss, 'dist_entropy', dist_entropy)

        # total_loss.backward()
        # self.optimizer_ac.step()



    def adaptive_entropy(self, episode):
        if episode < 3000:
           ada_ent = 0.001#- episode / 1000000
        else:
           ada_ent = 0.001
        # ada_ent = 1e-6
        #print('ada_ent', ada_ent)
        return ada_ent

    def get_full_state(self, robot):
        return FullState(robot.px, robot.py, robot.vx, robot.vy, robot.radius, robot.gx, robot.gy, robot.v_pref, robot.theta)
    # @profile
    #run k episodes and save the data in to the buffer
    def run_k_episodes(self, k, phase, update_memory=False, episode=None, print_failure=False, imitation_learning=False):
        self.robot1.policy.set_phase(phase)
        self.robot2.policy.set_phase(phase)
        self.robot3.policy.set_phase(phase)
        if not imitation_learning:
            self.robot1.policy.load_model(self.model['model0'])
            self.robot2.policy.load_model(self.model['model1']) 
            self.robot3.policy.load_model(self.model['model2'])

        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        success_time = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards_robot1 = []
        cumulative_rewards_robot2 = []
        cumulative_rewards_robot3 = []
        cumulative_rewards_total = []
        collision_cases = []
        timeout_cases = []



        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states_robot1 = []
            states_robot2 = []
            states_robot3 = []
            states = []
            actions_robot1 = []
            actions_robot2 = []
            actions_robot3 = []
            rewards_robot1 = []
            rewards_robot2 = []
            rewards_robot3 = []
            rewards_total = []
            dones = []
            #logging.info('start imitation_learning.')
            while not done:

                # action_robot1, action_robot2, action_robot1_indice, \
                # action_robot2_indice = self.policy.predict(ob)
                if imitation_learning:
                    action_robot1, action_robot2, action_robot3 = self.policy.predict(ob)
                else:
                    # logging.info('robot1 last state:%s.', ob.robot1_state)

                    action_robot1 = self.robot1.policy.predict(ob, 0)
                    action_robot2 = self.robot2.policy.predict(ob, 1)
                    action_robot3 = self.robot3.policy.predict(ob, 2)
                

                # print('action_indice', action_indice)
                #logging.info('start imitation_learning.')
                ob_next, reward, done, info = self.env.step(action_robot1, action_robot2, action_robot3)

                # state_robot1 = self.robot1.policy.last_state
                # logging.info('robot1 last state:%s.', self.robot1.policy.last_state)
                # logging.info('robots last state:%s.', self.policy.last_state)
                # states_robot1.append(state_robot1)
                # state_robot2 = self.robot2.policy.last_state
                # logging.info('robot2 last state:%s.', self.robot2.policy.last_state)
                # states_robot2.append(state_robot2)
                if imitation_learning:
                    state = ob_next
                    states.append(state)
                    # logging.info('state:%s.', state.robot1_state)
                else:
                    state = ob_next
                    states.append(state)

                # state_np = np.array(state)
                # states_np.append(state_np)

                actions_robot1.append(action_robot1)
                actions_robot2.append(action_robot2)
                actions_robot3.append(action_robot3)

                rewards_robot1.append(torch.tensor(reward[0], dtype=torch.float32, device=self.device))
                rewards_robot2.append(torch.tensor(reward[1], dtype=torch.float32, device=self.device))
                rewards_robot3.append(torch.tensor(reward[2], dtype=torch.float32, device=self.device))
                rewards_total.append(torch.tensor(reward[3], dtype=torch.float32, device=self.device))
                dones.append(done)

                ob = ob_next

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)
                    #logging.info('Danger')

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
                #logging.info('ReachGoal')
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
                #logging.info('Collision')
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
                #logging.info('Timeout')
            else:
                raise ValueError('Invalid end signal from environment')

            # visualize the training data
            #self.env.render('video', self.args.video_file)


            if update_memory:
                # trajectory_robot1_and_robot2 = []
                # for (state, action_robot1_, robot1_return_, action_robot2_, robot2_return_) \
                #         in list(zip(states, actions_robot1, rewards_robot1, actions_robot2, rewards_robot2)):
                #     trajectory_robot1_and_robot2.append([state, action_robot1_, robot1_return_, action_robot2_, robot2_return_])
                if imitation_learning:
                    #if not isinstance(info, Timeout):
                        #self.update_buffer(states, rewards_robot1, rewards_robot2, imitation_learning)
                    self.update_buffer(states, rewards_robot1, rewards_robot2, rewards_robot3, imitation_learning)
                else:
                    self.update_buffer(states, rewards_robot1, rewards_robot2, rewards_robot3, False)

            cumulative_reward_robot1 = sum([pow(self.gamma, t * self.robot1.time_step * self.robot1.v_pref)
                                            * reward for t, reward in enumerate(rewards_robot1)])
            cumulative_rewards_robot1.append(cumulative_reward_robot1)
            cumulative_reward_robot2 = sum([pow(self.gamma, t * self.robot2.time_step * self.robot2.v_pref)
                                            * reward for t, reward in enumerate(rewards_robot2)])
            cumulative_rewards_robot2.append(cumulative_reward_robot2)
            cumulative_reward_robot3 = sum([pow(self.gamma, t * self.robot3.time_step * self.robot3.v_pref )
                                            * reward for t, reward in enumerate(rewards_robot3)])
            cumulative_rewards_robot3.append(cumulative_reward_robot3)
            cumulative_rewards_total.append(cumulative_reward_robot1 + cumulative_reward_robot2 + cumulative_reward_robot3)
            cumulative_reward_total = cumulative_reward_robot1 + cumulative_reward_robot2 + cumulative_reward_robot3

            success_rate = success / k
            collision_rate = collision / k

            assert success + collision + timeout == (i+1)
            # avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.global_time
            avg_nav_time = self.env.global_time


            if imitation_learning == True:
                success_rate = success / (i+1)
                collision_rate = collision / (i+1)
                if print_failure:
                    extra_info = '' if episode is None else 'in episode {} '.format(episode)
                    logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, '
                                            'robot1 reward: {:.4f}, robot2 reward: {:.4f},robot3 reward: {:.4f}, total reward: {:.4f}'.
                                format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                                        cumulative_reward_robot1, cumulative_reward_robot2,cumulative_reward_robot3,
                                        cumulative_reward_total))

                if phase in ['val', 'test']:
                    total_time = sum(success_times + collision_times + timeout_times) * self.robot1.time_step
                    logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                                    too_close / total_time, average(min_dist))
            else:
                if print_failure:
                    extra_info = '' if episode is None else 'in episode {} '.format(episode)
                    logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, '
                                            'robot1 reward: {:.4f}, robot2 reward: {:.4f},robot3 reward: {:.4f}, total reward: {:.4f}'.
                                format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                                        cumulative_reward_robot1, cumulative_reward_robot2,cumulative_reward_robot3,
                                        cumulative_reward_total))

                if phase in ['val', 'test']:
                    total_time = sum(success_times + collision_times + timeout_times) * self.robot1.time_step
                    logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                                    too_close / total_time, average(min_dist))
            
            # if imitation_learning == True:# and isinstance(info, Timeout) == False:
            #     # self.trainer.optimize_batch(50, True)
            # if print_failure:
            #     logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            #     logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
####################################


            # if self.args.train_sil:
            #     #self.trainer.optimize_batch(self.train_batches)
            #     self._update_network(50)


    def update_target_model(self, target_model):
        self.model = copy.deepcopy(target_model)
        self.trainer.model = self.model



###################
# trainer for sil
###################
class Trainer(Explorer):
    def __init__(self, model, memory, device, batch_size, policy, num):
        self.network = model
        self.buffer = memory
        self.device = device
        self.batch_size = batch_size
        self.policy = policy
        self.num = num
        # some other parameters...
        #self.optimizer_sil = optim.Adam(self.network.parameters(), lr=1e-5)
        self.model = model
        self.batch_size = 100
        self.sil_beta = 0.1
        self.data_loader = None
        self.criterion = nn.MSELoss().to(device)
        self.optimizers = []
        self.optimizer_sils = []
        for i in range(num):
            keyname = 'model{}'.format(i)
            self.optimizer_sils.append(optim.Adam(self.network[keyname].parameters(), lr=1e-3))
            self.optimizers.append(optim.SGD(self.model[keyname].parameters(), lr=0.01, momentum=0.9))
        #self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.times = 0
    def set_learning_rate(self, learning_rate):
        #logging.info('Current learning rate in optimize_batch: %f', learning_rate)
        for i in range(self.num):
            keyname = 'model{}'.format(i)
            self.optimizers[i] = optim.SGD(self.model[keyname].parameters(), lr=learning_rate, momentum=0.9)

    def optimize_batch(self, num_batches, imitation_learning=False): # batch

        self.times = self.times + 1
        if self.data_loader is None:
                self.data_loader = []
                for i in range(self.num):
                    self.data_loader.append(DataLoader(self.buffer[i], self.batch_size, shuffle=True))  
        total_losses = []
        # losses = 0

        for i in range(len(self.data_loader)):
            losses = 0
            keyname = 'model{}'.format(int(i))
            for n in range(num_batches):


                states, robot_return = next(iter(self.data_loader[i])) # 对应关系？
                # mean_adv, num_valid_samples = 0, 0
                if states is not None:
                    if imitation_learning == True:
                        ada_ent = 0.001
                    else:
                        ada_ent = 0.001
                    # ada_ent = self.adaptive_entropy(self.times)
                    self.set_learning_rate(ada_ent)
                    self.optimizers[i].zero_grad()
                    #self.optimizer_sils[i].zero_grad()
                    states = torch.tensor(states, dtype=torch.float32)

                    robot_return = torch.tensor(robot_return, dtype=torch.float32)  # .unsqueeze(1)
                    # logging.info('%s',self.model[keyname].state_dict())
                    states = states.squeeze(0)
                    states = states.to(self.device)
                    value = self.model[keyname](states)

                    loss = self.criterion(value, robot_return)
                    #logging.info('%s',loss)
                    # logging.info('%s',self.model[keyname].state_dict())
                    # total_loss.backward()
                    loss.backward()
                    #logging.info('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx%s',self.model[keyname].state_dict())
                    self.optimizers[i].step()
                    #self.optimizer_sils[i].step()
                    #logging.info('%s',self.model[keyname].state_dict())
                    
                    losses += loss.data.item()
                average_loss = losses / num_batches
                #logging.debug('Average loss : %.2E', average_loss)
                total_losses.append(average_loss)
        return total_losses







