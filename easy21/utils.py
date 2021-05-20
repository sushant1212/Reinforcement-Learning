import numpy as np

class Easy21:
    colors = ["red", "black", "black"]

    def __init__(self, showing_card):
        self.first = showing_card
        self.sum = 0
        # adding the value of the first card to the dealer's sum
        self.sum += self.first
        # print("initial value:", self.sum)

    def reset(self):
        self.sum = self.first

    def step(self, s, a):
        """
        Input:
            s: agent's state which is the total sum             
            a (action): 0->hit ; 1->stick

        Output:
            s_prime: the new state
            r : reward: {0: draw or nothing has happened;  1: if the agent won; -1: if the agent lost} 
            val : {true: if the state s' is not terminal; false: if the state s' is a terminal state}
        """
        if(a == 0):
            # the agent has used "hit" action
            rand_no = np.random.randint(0, 3)
            new_no = np.random.randint(1,11)
            if(rand_no == 0):
                # print("red")
                new_no *= -1
            # else:
                # print("black")
            
            s_prime = s + new_no
            # print(new_no, "sum =", s_prime)
            
            r = 0
            val = True
            if(s_prime < 1 or s_prime > 21):
                r = -1
                val = False
            
            return (s_prime, r, val)
        
        elif(a == 1):
            # agent has sticked 
            r = 0
            while(self.sum < 17):
                (s_prime, r, val) = self.step(self.sum, 0)
                self.sum = s_prime
                if(val):
                    pass
                else:
                    break
            
            if(r == -1):
                # dealer has lost the game
                return(s, 1, False)
            
            else:
                if(self.sum > s):
                    return (s, -1, False)
                elif(self.sum == s):
                    return (s, 0, False)
                else:
                    return (s, +1, False)