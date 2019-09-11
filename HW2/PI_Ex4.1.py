#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


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


# In[4]:


def actionNumb(i,j):
    if((i==3 and j==0)):
        return [0,1,2]
    elif((i==0 and j==3)):
        return [1,2,3]
    else:
        return [0,1,2,3]


# In[7]:


GridVal = np.zeros((4,4))
#Making a Policy Function
prev = 0
count=0
police = np.ones((4,4,4))/4
policy_stable=False
ValueLog =[]
ValueLog.append(GridVal)
while(policy_stable==False):
    for k in range(1000):
        diff = 0
        TempVal = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                #We will use V* formula which maximizes over all actions
    #             val = []
                for act in range(4):
                    #As for a given state and action the next state-
                    # and reward are fixed so P(s',r|s,a)=1
                    Next_sx,Next_sy,reward = rew(i,j,act)
    #                 val.append(reward+gamma*GridVal[Next_sy][Next_sx])
                    TempVal[j][i] += police[j][i][act]*(reward+GridVal[Next_sy][Next_sx])

    #                 print(1)
        mod = np.abs(np.linalg.det(GridVal-TempVal))
        GridVal = TempVal
    #     print(mod)
        if(mod<0.0000000001):
            if(prev==1):
#                 print("why")
                break
            else:
                prev = 1
        else:
            prev = 0
    #Policy Improvement
    ValueLog.append(TempVal)
    print(TempVal)
    policy_stable = True
    New_policy = police.copy()
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
            CurrentBestAct = np.argmax(police[j][i])
            if(BestAct!=CurrentBestAct):
                policy_stable = False
#                 print("F")
            police[j][i] = np.zeros(4)
            police[j][i][BestAct]=1
    Diff = 0
    for k in range(4):
        Diff+=np.abs(np.linalg.det(New_policy[:][:][k])-np.linalg.det(police[:][:][k]))
            
    if(Diff<0.00001):
        count+=1
    else:
        count=0
    if(count>3):
        break
        
# GridVal


# In[ ]:





# In[ ]:





# In[ ]:




