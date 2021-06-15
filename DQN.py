import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import deque
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###### HYPERPARAMETERS############
LR = 0.01
UPDATE_EVERY = 10
BUFFER_SIZE = 2000
BATCH_SIZE = 64
GAMMA = 0.99
NUM_EPISODES = 1000
EPSILON = 0.1
##################################


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__() 
        self.linear_stack = nn.Sequential(
            nn.Linear(4,50),
            nn.ReLU(),
            nn.Linear(50,2)
        )
        
    def forward(self, x):
        x = self.linear_stack(x)
        return x    

class ExperienceReplay:
    def __init__(self, max_capacity) -> None:
        self.list = deque(maxlen = max_capacity)

    def insert(self, val:tuple) -> None:
        assert len(val) == 5
        self.list.append(val)

    def __len__(self):
        return len(self.list)

    def sample_batch(self, batch_size):
        l = []
        for i in range(batch_size):
            ind = np.random.randint(0, len(self.list))
            l.append(self.list[ind])
        
        return l

class Agent:
    def __init__(self) -> None:
        self.model = DQN().to(device)
        self.fixed_targets = DQN().to(device)
        self.env = gym.make('CartPole-v1')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.expreplay = ExperienceReplay(int(BUFFER_SIZE))
        self.state = self.env.reset()
        self.gamma = GAMMA
        self.total_reward = 0
        self.steps = 0

    def sample_action(self, epsilon, current_state):
        # samples action in an epsilon greedy policy
        if(np.random.random() < epsilon):
            a = self.env.action_space.sample()
            (next_state, reward, done, _) = self.env.step(a)
            return (next_state, reward, done, _, a)
        else:
            cs = torch.from_numpy(current_state).reshape(1,4).float().to(device)
            a = torch.argmax(self.model(cs))
            (next_state, reward, done, _) = self.env.step(a.item())
            return (next_state, reward, done, _, a)
    
    def train_loop(self):
        (next_state, reward, done, _, a) = self.sample_action(EPSILON,self.state)
        self.total_reward += reward
        self.expreplay.insert((self.state,a,reward,next_state,done))
        l = self.expreplay.sample_batch(BATCH_SIZE)
        for t in l:
            (state, act, ri, ns, d) = t
            target = None
            if(d == True):
                target = torch.tensor(ri, device=device)
            else:
                target = torch.tensor(ri, device=device) + self.gamma * torch.max(self.fixed_targets(torch.from_numpy(ns).reshape(1,4).float().to(device)))

            Q = self.model(torch.from_numpy(state).reshape(1,4).float().to(device))[0][act]

            loss = self.loss_fn(target, Q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.steps += 1
        if (done == True):
            self.state = self.env.reset()
        else:
            self.state = next_state
        
        if(self.steps%UPDATE_EVERY == 0):
            self.fixed_targets.load_state_dict(self.model.state_dict())



        return done

if __name__ == "__main__":
    agent = Agent()
    for i in range(NUM_EPISODES):
        var = False
        while(not var):
            var = agent.train_loop()
            agent.env.render()
        print(f"Number of Episodes : {i+1}   Total reward = {agent.total_reward}" )
        agent.total_reward = 0
        if i > 100:
            EPSILON = 0.01
        