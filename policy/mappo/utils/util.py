import copy
import numpy as np

import torch
import torch.nn as nn
import time

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def transform(robot_states, human_states, human_num, att_agents):
    """  
    robot_state:[batch, robot_obs_dim]   
    human_state:[batch, human_num, human_obs_dim]
    robot_obs:[px, py, gx, gy, v]
    human_obs:[px, py, vx, vy, theta]
    """
    robot_states = check(robot_states)
    # Allow disabling human observations by passing human_num == 0 or empty tensors.
    if (human_num is None) or (human_num <= 0) or (human_states is None):
        # Return an empty tensor on the same device/dtype as robot_states.
        device = robot_states.device if hasattr(robot_states, "device") else torch.device("cpu")
        dtype = robot_states.dtype if hasattr(robot_states, "dtype") else torch.float32
        batch = robot_states.shape[0]
        # keep last dim as-is if possible; otherwise default to 0
        human_obs_dim = 0
        if human_states is not None and hasattr(human_states, "shape") and len(human_states.shape) == 3:
            human_obs_dim = human_states.shape[2]
        return torch.empty((batch, 0, human_obs_dim), device=device, dtype=dtype)

    if human_num < att_agents:
        human_states = human_states[:, :human_num, :]
    human_states = check(human_states).to(device=robot_states.device, dtype=torch.float32)

    robot_states = robot_states.unsqueeze(1)
    human_states[:, :, 0] -= robot_states[:, :, -2]
    human_states[:, :, 1] -= robot_states[:, :, -1]
    
    # calculate atan2
    human_states[:, :, 4] = torch.atan2(human_states[:, :, 1], human_states[:, :, 0])

    return human_states #new_state:[batch, human_num, human_obs_dim]


