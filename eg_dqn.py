import gym
import numpy as np
import math
import sys
import torch
import torch.nn as nn
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import faulthandler
faulthandler.enable()

# print(sys.argv[1])
file_name = 'RL_data_dqn'  # sys.argv[1]
state_data = data = np.zeros((1, 4), dtype=int)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, _input, hidden, output):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(_input, hidden)
        self.l2 = nn. Linear(hidden, output)
        # self.l3 = nn. Linear(hidden, output)

        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        return x.view(x.size(0), -1)


def get_epsilon(i): return max(
    0.01, min(1, 1.0 - math.log10((i+1)/25)))  # epsilon-greedy; 隨時間遞減


def get_lr(i): return max(
    0.01, min(0.5, 1.0 - math.log10((i+1)/25)))  # learning rate; 隨時間遞減


def select_action(state, i_episode):
    global steps_done
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)

    eps_threshold = get_epsilon(i_episode)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def get_state(observation, n_buckets, state_bounds):
    state = [0] * len(observation)
    for i, s in enumerate(observation):  # 每個 feature 有不同的分配
        l, u = state_bounds[i][0], state_bounds[i][1]  # 每個 feature 值的範圍上下限
        if s <= l:  # 低於下限，分配為 0
            state[i] = 0
        elif s >= u:  # 高於上限，分配為最大值
            state[i] = n_buckets[i] - 1
        else:  # 範圍內，依比例分配
            # state[i] = int(((s - l) / (u - l)) * n_buckets[i])
            state[i] = int(round(((s - l) / (u - l)) * (n_buckets[i]-1), 0))
    global state_data
    state_data = np.append(state_data, [state], axis=0)
    with open('sate_data.npy', 'wb') as f:
        np.save(f, state_data)
    
    return torch.tensor(state).unsqueeze(0)




# gamma = 0.99  # reward discount factor
reward_data = np.array([])

# Q-learning
Gamma = [0.99]


BATCH_SIZE = 8
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 4
LR = 0.01

# 準備 Q table
## Environment 中各個 feature 的 bucket 分配數量
## 1 代表任何值皆表同一 state，也就是這個 feature 其實不重要
n_buckets = (1, 1, 6, 3)

env = gym.make('CartPole-v0')

n_actions = env.action_space.n  # Action 數量
policy_net = DQN(4, 128, n_actions).to(device)
target_net = DQN(4, 128, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

## State 範圍
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(1000)
steps_done = 0
episode_durations = []


for i_episode in range(300):
    epsilon = get_epsilon(i_episode)
    lr = get_lr(i_episode)

    observation = env.reset()
    rewards = 0
    state = get_state(observation, n_buckets, state_bounds)  # 將連續值轉成離散
    
    for t in range(250):
        env.render()

        action = select_action(state, i_episode)
        observation, reward, done, info = env.step(action.item())
        
        reward = torch.tensor([reward], device=device)

        rewards += reward.item()
        
        if not done:
            next_state = get_state(observation, n_buckets, state_bounds)
        else:
            next_state = None
        
        memory.push(state, action, next_state, reward)

        # 前進下一 state
        state = next_state

        optimize_model()
        
        
        if done:
            print('Episode finished after {} timesteps, total rewards {}, i_episode {}'.format(
                t+1, rewards, i_episode))

            episode_durations.append(t + 1)
            # plot_durations()
            

            reward_data = np.append(reward_data, [rewards], axis=0)
            with open(file_name, 'wb') as f:
                np.save(f, reward_data)
            break
    
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


print('Complete')
env.close()



