#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#The Reward,Next State for give s,a
def rew(sx,sy,a):
    reward = 0
    if((sx==1 and sy==0)):
        sy=4
        reward = 10
    elif(sx==3 and sy==0):
        sy=2
        reward = 5
    elif(a==1):
        if((sx-1)<0):
            reward = -1
        else:
            sx-=1
    elif(a==2):
        if((sx+1)>4):
            reward = -1
        else:
            sx+=1
    elif(a==3):
        if((sy-1)<0):
            reward = -1
        else:
            sy-=1
    elif(a==0):
        if((sy+1)>4):
            reward = -1
        else:
            sy+=1
    return sx,sy,reward
        


# In[3]:


#Calculating the Reward Matrix & Probability Matrix
GridVal = np.zeros((5,5)) #For Rewards
TempVal = np.zeros((25,25))
for i in range(5):
    for j in range(5):
        if((j==0 and i==1) or (j==0 and i==3)):
            for act in range(1):
                Next_sx,Next_sy,reward = rew(i,j,act)
                GridVal[j][i] += (reward)
                IndexInProbMatJ = Next_sy*5 + Next_sx
                IndexInProbMatI = j*5 + i
                TempVal[IndexInProbMatI][IndexInProbMatJ] += 1
                print("samething")
                print(IndexInProbMatI)
                print(IndexInProbMatJ)
        else:
            for act in range(4):
                Next_sx,Next_sy,reward = rew(i,j,act)
                GridVal[j][i] += 0.25*(reward)
                IndexInProbMatJ = Next_sy*5 + Next_sx
                IndexInProbMatI = j*5 + i
                TempVal[IndexInProbMatI][IndexInProbMatJ] += 0.25

Rewards = (GridVal).flatten()
Prob = TempVal
gamma = 0.9
X = (np.identity(25) - (gamma*Prob)) # I-gammaP
Value = np.matmul(np.linalg.inv(X),Rewards)
print(Rewards)
print(TempVal)
print(Value)
# Value
# Prob


# In[4]:


Value.reshape((5,5))


# In[ ]:




