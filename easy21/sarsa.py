from utils import Easy21
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from monte_carlo import Agent_MC

class Agent:
    def __init__(self, env, start_card, lambd):
        self.Q = np.zeros((21, 2), dtype=np.float32)
        self.V = np.zeros((21,1), dtype=np.float32)
        self.N = np.zeros((21,2), dtype=np.int32) # N(s,a)
        self.sum = start_card
        self.epsilon = 1.0
        self.env = env
        self.gamma = 1
        self.N0 = 100
        self.lambd = lambd
        self.N = np.zeros((21,2))
        self.E = np.zeros((21,2))

    def sarsa(self):
        val = True
        choices = [0, 1]
        prob = [self.epsilon/2, (self.epsilon/2) +1-self.epsilon]
        choice = np.random.choice(choices, p=prob)

        # generating the first action 
        action = 0
        if(choice == 0): # exploration
            action = np.random.randint(0,2)
        
        else: # exploitation
            action = np.argmax(self.Q[self.sum-1, :]) # greedily choose the action 


        while(val):           
            # incrementing the N(s,a) value by 1
            self.N[self.sum-1, action] += 1
            
            self.E[self.sum-1, action] += 1 # updating the E value for current state and action 
                        
            
            # taking step in env
            prev_state = self.sum
            (self.sum, r, val) = env.step(self.sum, action)
           
            # generate next state's action using current policy which is epsilon greedy. 
            # REMARK: If we were doing q learning here, this action would have been taken greedily, to estimate Q(s',a').
            #         After that, action is taken according to epsilon greedy
            
            prev_action = action

            if(val == False): # epsiode is going to end after this iteration. State values may be greater than 21
                delta = r  - self.Q[prev_state-1, prev_action]
                factor  = float(1/(self.N[prev_state-1, prev_action]))* delta
                self.Q +=  factor * self.E 
                continue
                
            choices = [0, 1]
            prob = [self.epsilon/2, (self.epsilon/2) +1-self.epsilon]
            choice = np.random.choice(choices, p=prob)
            action = 0
            if(choice == 0): # exploration
                action = np.random.randint(0,2)
            else: # exploitation
                action = np.argmax(self.Q[self.sum-1, :]) # greedily choose the action 


            # online update of Q values
            delta = r + self.gamma*self.Q[self.sum-1, action] - self.Q[prev_state-1, prev_action]
            factor  = float(1/(self.N[prev_state-1, prev_action]))* delta
            self.Q +=  factor * self.E 

            # updating epsilon
            n = self.N[prev_state-1,0]+self.N[prev_state-1,1]
            self.epsilon = float(self.N0)/float(self.N0 + n)

            # updating E values
            self.E *= (self.lambd*self.gamma)

            
        # episode has ended
        return self.Q
    
    
if __name__ == "__main__":
    dealer_card = np.random.randint(1,12)
    env = Easy21(dealer_card)
    
    # training on MC to get the Q* values. Note that we are saying that this is Q* because MC has low bias.
    Q_star = np.zeros((21,2))

    player_card = np.random.randint(1, 11)
    agent_mc = Agent_MC(env, player_card)
    for _ in range(50000): 
        Q_star = agent_mc.monte_carlo()
        agent_mc.G = []
        agent_mc.state_actions=[]
        agent_mc.epsilon=1.0
        agent_mc.sum=np.random.randint(1, 11)
        env.reset()

    print("Completed evaluating Q* values using Monte Carlo Learning")

    # learning using sarsa(lambda)

    lambds = np.arange(0, 1.1, 0.1)
    mse = []
    for i in range(len(lambds)):
        lambd = lambds[i]
        agent = Agent(env, np.random.randint(1,11), lambd)
        for _ in range(1000):
            Q = agent.sarsa()
            agent.sum = np.random.randint(1,11)
            agent.epsilon = 1.0
            agent.env.reset()
        mse.append(np.sum(np.square(Q_star-Q)))

    # print(Q)
    temp = list(np.arange(0, 1.1, 0.1))
    fig1,ax1 = plt.subplots()
    ax1.plot(temp, mse)
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("Mean Squared Error")
        
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("episodes")
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_ylim((0,20))
    
    mse1 = []
    mse2 = []
    mse3 = []
    mse4 = []
    mse5 = []

    x = list(np.arange(1, 100001))
    agent = Agent(env, np.random.randint(1,11), 0.0)
    for _ in range(100000):
        Q = agent.sarsa()
        agent.sum = np.random.randint(1,11)
        agent.epsilon = 1.0
        agent.env.reset()
        mse1.append(np.sum(np.square(Q_star-Q)))
    
    agent = Agent(env, np.random.randint(1,11), 0.9)
    for _ in range(100000):
        Q = agent.sarsa()
        agent.sum = np.random.randint(1,11)
        agent.epsilon = 1.0
        agent.env.reset()
        mse2.append(np.sum(np.square(Q_star-Q)))
    
    agent = Agent(env, np.random.randint(1,11), 0.75)
    for _ in range(100000):
        Q = agent.sarsa()
        agent.sum = np.random.randint(1,11)
        agent.epsilon = 1.0
        agent.env.reset()
        mse3.append(np.sum(np.square(Q_star-Q)))
    
    agent = Agent(env, np.random.randint(1,11), 0.5)
    for _ in range(100000):
        Q = agent.sarsa()
        agent.sum = np.random.randint(1,11)
        agent.epsilon = 1.0
        agent.env.reset()
        mse4.append(np.sum(np.square(Q_star-Q)))
    
    agent = Agent(env, np.random.randint(1,11), 0.3)
    for _ in range(100000):
        Q = agent.sarsa()
        agent.sum = np.random.randint(1,11)
        agent.epsilon = 1.0
        agent.env.reset()
        mse5.append(np.sum(np.square(Q_star-Q)))

    ax2.plot(x,mse1,label="lambda=0", color='r')
    ax2.plot(x,mse2,label="lambda=0.9", color='g')
    ax2.plot(x,mse3,label="lambda=0.75", color='c')
    ax2.plot(x,mse4,label="lambda=0.5", color='b')
    ax2.plot(x,mse5,label="lambda=0.3", color='m')

    plt.legend()
    plt.show()

    