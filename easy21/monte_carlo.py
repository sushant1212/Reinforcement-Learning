from utils import Easy21
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

class Agent_MC:
    def __init__(self, env, start_card):
        self.Q = np.zeros((21, 2), dtype=np.float32)
        self.V = np.zeros((21,1), dtype=np.float32)
        self.N = np.zeros((21,2), dtype=np.int32) # N(s,a)
        self.sum = start_card # self.sum represents the current state
        self.epsilon = 1.0
        self.env = env
        self.gamma = 1.0
        self.N0 = 100
        self.G = [] # list of returns in the whole episode
        self.state_actions = [] # state action pairs 

    def monte_carlo(self):
        val = True

        while(val):
            choices = [0, 1]
            prob = [self.epsilon, (1-self.epsilon)]
            choice = np.random.choice(choices, p=prob)

            action = 0
            if(choice == 0): # exploration
                action = np.random.randint(0,2)
            
            else: # exploitation
                action = np.argmax(self.Q[self.sum-1, :]) # greedily choose the action 
            
            self.N[self.sum-1, action] += 1
            self.state_actions.append((self.sum-1, action))
            
            # updating epsilon
            n = self.N[self.sum-1,0]+self.N[self.sum-1,1]
            self.epsilon = float(self.N0)/float(self.N0 + n)

            # taking step in env
            (self.sum, r, val) = self.env.step(self.sum, action)
            self.G.append(r)
            l = len(self.G)
            l -= 2
            gamma = self.gamma
            
            # updating G values
            while(l>=0):
                self.G[l] += gamma*r
                gamma *= self.gamma
                l-=1

        # episode has ended
        for i in range(len(self.G)):
            st = self.state_actions[i][0]
            act = self.state_actions[i][1]
            self.Q[st, act] += ((1.0/float(self.N[st, act]))*(self.G[i]-self.Q[st,act]))

        return self.Q
    
    
if __name__ == "__main__":

    arr = np.zeros((10,21))
    for dealer_card in range(1,11):
    # dealer_card = 21
        env = Easy21(dealer_card)
        player_card = np.random.randint(1, 11)
        agent = Agent_MC(env, player_card)
        for _ in range(100000): # 1000 episodes
            Q = agent.monte_carlo()
            # if(_%50000 == 0):
            #     print("episode", _, "dealer_card", dealer_card)
            agent.G = []
            agent.state_actions=[]
            agent.epsilon=1.0
            agent.sum=np.random.randint(1, 11)
            agent.env.reset()
        print(f"Q for dealer card = ", {dealer_card})
        print(Q)
        agent.Q = np.zeros((21,2))
        V = np.max(Q, axis=1)
        arr[dealer_card-1,:] = V
  
    

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.arange(1,22,1)
    Y = np.arange(1,11,1)
    X,Y = np.meshgrid(X,Y)
    Z = arr
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('player sum')
    ax.set_ylabel('dealer showing')
    ax.set_zlabel('reward')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    