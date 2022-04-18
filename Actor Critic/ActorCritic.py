# Environment is Lunar Lander
from gym import spaces
from gym.spaces import space
from torch.nn.modules.activation import ReLU, Softmax
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as c
from collections import deque
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################
LR = 0.00005
NUM_EPISODES = 10000
GAMMA = 0.99
###############################################

# class Actor(nn.Module):
#     def __init__(self):
#         super(Actor,self).__init__()
#         self.linear_stack = nn.Sequential(
#             nn.Linear(8,10),
#             nn.ReLU(),
#             nn.Linear(10,10),
#             nn.ReLU(),
#             nn.Linear(10,4),
#             nn.Softmax()
#         )
#     def forward(self, x):
#         x = self.linear_stack(x)
#         return x

# class Critic(nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.linear_stack = nn.Sequential(
#             nn.Linear(8,10),
#             nn.ReLU(),
#             nn.Linear(10,10),
#             nn.ReLU(),
#             nn.Linear(10,4)
#         )

#     def forward(self, x):
#         x = self.linear_stack(x)
#         return x

class Actor_critic(nn.Module):
    def __init__(self):
        super(Actor_critic,self).__init__()
        self.actor_linear_stack = nn.Sequential(
            nn.Linear(8,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,4),
            nn.Softmax()
        )
        self.critic_linear_stack = nn.Sequential(
            nn.Linear(8,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        actor = self.actor_linear_stack(x)
        critic = self.critic_linear_stack(x)
        return actor, critic

class Agent:
    def __init__(self) -> None:
        # self.actor = Actor().to(device)
        # self.critic = Critic().to(device)
        self.model = Actor_critic().to(device)
        self.env = gym.make('LunarLander-v2')
        self.state = self.env.reset()
        # self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=LR)
        # self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=LR)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.total_rewards = 0
    
    def train_loop(self):
        probs, v = (self.model((torch.from_numpy(self.state).unsqueeze(0).float().to(device))))
        probs = torch.squeeze(probs)
        m = c.Categorical(probs)
        action = m.sample()
        next_state, reward, done, _ = self.env.step(action.item())
        self.env.render()
        _, v_prime = self.model(torch.from_numpy(next_state).unsqueeze(0).float().to(device))
        # probs2 = torch.squeeze(probs2)
        # q = self.critic(torch.from_numpy(self.state).unsqueeze(0).float().to(device))
        # probs2 = (self.actor((torch.from_numpy(next_state).unsqueeze(0)).float().to(device)))
        # m2 = c.Categorical(probs2)
        # action2 = m2.sample().item()
        if done == True:
            v_prime[0] = 0
        td_error = reward + GAMMA*v_prime[0] - v[0]
        loss_a = -m.log_prob(action) * td_error
        loss_obj = nn.MSELoss()
        loss_c = loss_obj(torch.tensor([reward], device=device).float() + GAMMA*v_prime[0].float(), v[0].float())
        loss = loss_a + loss_c
        # self.optimizer_actor.zero_grad()
        # self.optimizer_critic.zero_grad()
        self.optimizer.zero_grad()
        # loss_a.backward()
        # loss_c.backward()
        loss.backward()
        # self.optimizer_actor.step()
        # self.optimizer_critic.step()
        self.optimizer.step()

        # update next state
        if done == True:
            self.state = self.env.reset()
        else:
            self.state = next_state
        return done, reward  

if __name__ == "__main__":
    agent = Agent()
    for i in range(NUM_EPISODES):
        done = False
        total_reward = 0
        while not done:
            done, reward = agent.train_loop()
            total_reward += reward
        writer.add_scalar("reward", total_reward, i)
        writer.flush()
        print(f"Number of Episodes : {i+1}   Total reward = {total_reward}")
    writer.close()