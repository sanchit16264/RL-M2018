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


#Making a 5x5 Grid
GridVal = np.zeros((5,5))
#Gamma Value give as 0.9
gamma = 0.9
for k in range(1000):
    TempVal = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            #We will use V* formula which maximizes over all actions
            val = []
            if((j==1 and i==0) or (j==3 and i==0)):
                for act in range(1):
                    #As for a given state and action the next state-
                    # and reward are fixed so P(s',r|s,a)=1
                    Next_sx,Next_sy,reward = rew(i,j,act)
                    val.append(reward+gamma*GridVal[Next_sy][Next_sx])
            else:
                for act in range(4):
                    #As for a given state and action the next state-
                    # and reward are fixed so P(s',r|s,a)=1
                    Next_sx,Next_sy,reward = rew(i,j,act)
                    val.append(reward+gamma*GridVal[Next_sy][Next_sx])
            TempVal[j][i] = np.max(val)
    GridVal = TempVal
GridVal
                
                


# In[ ]:




