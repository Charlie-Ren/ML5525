#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
from matplotlib import pyplot as plt
A=np.random.random((20,10))
b=np.random.rand(20)


# In[34]:


def lsq_iter(A,b):
    w1=np.zeros(10) #w1=0 and also use for wk"
    ws=np.dot(np.linalg.pinv(A),b) #"wstar"
    miu=1/np.square(np.linalg.norm(A)) #"miu"
    l=list()
    for i in range(0,500):
        wt=w1-miu*np.dot(np.transpose(A),(np.dot(A,w1)-b)) #"equation"
        temp=np.linalg.norm(wt-ws)
        l.append(temp)
        if temp<=0.01:
            print ("we are done")
            break
        w1=wt
    return l


# In[36]:


res=lsq_iter(A,b)
plt.plot(res)
plt.xlabel("times of iteration")
plt.ylabel("wk - w")
plt.show()


# In[28]:


def lsq(A,b): #closed form
    para1=np.dot(np.transpose(A),A)
    para1=np.linalg.inv(para1)
    para2=np.dot(para1,np.transpose(A))
    w=np.dot(para2,b) #（ATA）-1 AT b
    return w


# In[29]:


tt=lsq(A,b)
print (tt)

