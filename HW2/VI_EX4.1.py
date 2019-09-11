#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def rew(sx,sy,a):
    reward = -1
    if((sx==0 and sy==0) or (sx==3 and sy==3)):
        return sx,sy,0
    elif(a==1):
        if((sx-1)<0):
            reward = -1
        else:
            sx-=1
    elif(a==2):
        if((sx+1)>3):
            reward = -1
        else:
            sx+=1
    elif(a==3):
        if((sy-1)<0):
            reward = -1
        else:
            sy-=1
    elif(a==0):
        if((sy+1)>3):
            reward = -1
        else:
            sy+=1
    return sx,sy,reward


# In[12]:


GridVal = np.zeros((4,4))
police = np.zeros((4,4,4))
count = 0
for k in range(1000):
    TempVal = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            #We will use V* formula which maximizes over all actions
            val = []
            for act in range(4):
                #As for a given state and action the next state-
                # and reward are fixed so P(s',r|s,a)=1
                Next_sx,Next_sy,reward = rew(i,j,act)
                val.append(reward+GridVal[Next_sy][Next_sx])
            TempVal[j][i] = np.max(val)
    diff = mod = np.abs(np.linalg.det(GridVal)-np.linalg.det(TempVal))
    print(GridVal)
    GridVal = TempVal
    if(diff<0.0001):
        count+=1
    else:
        count=0
    if(count>4):
        break
# print(GridVal)


# In[11]:


#Now we find the Policy
for i in range(4):
    for j in range(4):
        val = []
        for act in range(4):
            #As for a given state and action the next state-
            # and reward are fixed so P(s',r|s,a)=1
            Next_sx,Next_sy,reward = rew(i,j,act)
#                 val.append(reward+gamma*GridVal[Next_sy][Next_sx])
            val.append((reward+GridVal[Next_sy][Next_sx]))
        BestAct = np.argmax(val)
        police[j][i][BestAct] = 1
police


# In[ ]:




