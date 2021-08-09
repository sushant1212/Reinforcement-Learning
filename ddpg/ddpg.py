from numpy.core.fromnumeric import mean
from numpy.lib.function_base import copy
from numpy.lib.histograms import _hist_bin_fd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Tanh
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np
from optim_env import OptimEnv
from collections import deque
# torch.autograd.set_detect_anomaly(True)
import gym
import copy

"""
Reference : https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py
"""

###### HYPERPARAMS #########
MAX_CAPACITY = 1000000
GAMMA = 0.99
TAU = 0.995
LR = 0.001
BATCH_SIZE = 128
UPDATE_EVERY = 50
NUM_EPISODES = 10000
UPDATE_AFTER = 1000
START_STEPS = 5000
NOISE_SCALE = 0.1
############################

class ExperienceReplay:
    def __init__(self, obs_dim, act_dim, size=MAX_CAPACITY) -> None:
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32) # s
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32) # s'
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32) # a
        self.rew_buf = np.zeros(size, dtype=np.float32) # r
        self.done_buf = np.zeros(size, dtype=np.float32) # done
        self.ptr, self.size, self.max_size = 0, 0, size
        self.size=0
        self.max_size=MAX_CAPACITY

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=BATCH_SIZE):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



class Actor(nn.Module): # Policy Network
    def __init__(self, obs_dim, hidden_layer_size, act_dim, act_max):
        super(Actor, self).__init__()
        self.act_max = act_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        x = x * self.act_max
        return x

class Critic(nn.Module): # Q-Network
    def __init__(self, obs_dim, hidden_layer_size, act_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear((obs_dim+act_dim), hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        x = self.net(x)
        x = torch.squeeze(x, dim=-1)
        return x

class Agent:

    def __init__(self) -> None:
        # Initializing the environment
        self.env = gym.make('Pendulum-v0')
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.act_max = self.env.action_space.high[0]

        # Initializing the networks
        self.policy = Actor(self.obs_dim, 15, self.act_dim, self.act_max)
        self.target_policy = copy.deepcopy(self.policy)
        for p in self.target_policy.parameters():
            p.requires_grad = False
        self.q = Critic(self.obs_dim, 15, self.act_dim)
        self.target_q = copy.deepcopy(self.q)
        for p in self.target_q.parameters():
            p.requires_grad = False
        
        # Experience Replay
        self.replay_buffer = ExperienceReplay(self.obs_dim, self.act_dim, MAX_CAPACITY)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=LR)

        # total rewards
        self.total_rewards = 0
        self.state = self.env.reset()
        self.total_steps = 0
        self.max_ep_len = 200


    def compute_loss_q(self, data):
        s, a, r, s_prime, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.q(s, a)
        with torch.no_grad():
            q_target = self.target_q(s_prime, self.target_policy(s_prime))
            backup = r + GAMMA * (1-done) * q_target
        
        loss_q = ((q-backup)**2).mean()
        return loss_q

    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.q(o, self.policy(o))
        return -q_pi.mean()

    def update(self, data):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.q.parameters():
            p.requires_grad = False
        
        self.policy_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.policy_optimizer.step()

        for p in self.q.parameters():
            p.requires_grad = True
        
        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.target_q.parameters()):
                p_targ.data.mul_(TAU)
                p_targ.data.add_((1-TAU) * p.data)
            
            for p, p_targ in zip(self.policy.parameters(), self.target_policy.parameters()):
                p_targ.data.mul_(TAU)
                p_targ.data.add_((1-TAU) * p.data)
    
    def get_action(self, o, noise_scale):
        a = self.policy(torch.as_tensor(o, dtype=torch.float32))
        a = a.detach().numpy()
        a += noise_scale * np.random.randn(self.act_dim)
        return a

    def train_loop(self):
        d = False
        ep_len = 0
        while(d == False):
            if(self.total_steps > START_STEPS):
                a = self.get_action(self.state, noise_scale=NOISE_SCALE)
            else:
                a = self.env.action_space.sample()

            o2, r, d, _ = self.env.step(a)
            self.total_rewards += r
            ep_len += 1
            self.total_steps += 1
            d = False if ep_len == self.max_ep_len else d
            self.replay_buffer.store(self.state, a, r, o2, d)

            ### VISUALIZING ###
            self.env.render()
            ###################            

            self.state = o2
            if d or (ep_len == self.max_ep_len):
                d = True
                self.state = self.env.reset()
            
            if self.total_steps > UPDATE_AFTER and self.total_steps % UPDATE_EVERY == 0:
                for _ in range(UPDATE_EVERY):
                    batch = self.replay_buffer.sample_batch(BATCH_SIZE)
                    self.update(data=batch)

            
if __name__ == "__main__":
    agent = Agent()
    
    for i in range(NUM_EPISODES):
        agent.train_loop()
        total_reward = agent.total_rewards
        agent.total_rewards = 0
        writer.add_scalar("reward", total_reward, i)
        writer.flush()
        print(f"Number of Episodes : {i+1}   Total reward = {total_reward}" )
        # print()
        
        # print('\n\n')
    
    writer.close()
