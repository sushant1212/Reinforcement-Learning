# Environment is LunarLander-v2
from gym import spaces
from gym.spaces import space
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as c
# torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################
LR = 0.01
NUM_EPISODES = 10000
GAMMA = 0.99
###############################################


class Reinforce(nn.Module):
    def __init__(self):
        super(Reinforce,self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(8,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            # nn.Linear(50,50),
            # nn.ReLU(),
            nn.Linear(10,4),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.linear_stack(x)
        return x

class Agent:
    def __init__(self) -> None:
        self.model = Reinforce().to(device)
        self.env = gym.make('LunarLander-v2')
        self.state = self.env.reset()
        self.episode = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.G = []
        self.losses = []
        self.total_rewards = 0

    def sample_episode(self):
        self.total_rewards = 0
        self.losses = []
        done = False
        state = self.state
        while done==False:
            probs = (self.model((torch.from_numpy(state).unsqueeze(0)).float().to(device)))
            m = c.Categorical(probs)
            action = m.sample()
            next_state, reward, done, _ = self.env.step(action.item())
            self.episode.append([state,action,reward])
            self.losses.append(-(m.log_prob(action)))
            self.env.render()
            state = next_state
            self.total_rewards += reward   
        self.G = []
        for i in range(len(self.episode)):
            self.G.append(0)
            for j in range(i,len(self.episode)):
                _,_,r = self.episode[j]
                self.G[-1] += (GAMMA**(j-i))*r

    def train_loop(self):
        self.sample_episode()
        loss = 0
        for i in range(len(self.episode)):
            gt = self.G[i]
            li = self.losses[i]
            loss +=  li * gt
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.state = self.env.reset()
        self.episode = []
        return self.total_rewards

if __name__ == "__main__":
    agent = Agent()
    for i in range(NUM_EPISODES):
        total_reward = agent.train_loop()
        print(f"Number of Episodes : {i+1}   Total reward = {total_reward}" )