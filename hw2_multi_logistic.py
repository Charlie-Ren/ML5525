#!/usr/bin/env python
# coding: utf-8

# In[45]:


global k
global batchsize
batchsize=120 # 720 data, so I used 120.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data_1=pd.read_csv("mfeat_train.csv",index_col=0).to_numpy()
data_2=pd.read_csv("mfeat_test.csv",index_col=0).to_numpy()
[m,n]=data_1.shape
print (m,n)
data_1[:,n-1]-=1
data_2[:,n-1]-=1
train_y=data_1[:,n-1].reshape(-1,1)
train_x=data_1[:,:n-1]
test_y=data_2[:,n-1].reshape(-1,1)
test_x=data_2[:,:n-1]
k=len(np.unique(train_y)) #num of category 10


# In[46]:


def softmax(X): #m*10
    for i in range(len(X)):
        X[i,:]-=np.max(X[i,:])
        X[i,:]=np.exp(X[i,:])/np.sum(np.exp(X[i,:]))
    return X #NP


# In[47]:


def batch_data(X,y,itr): #itr means which block is neede in this function
    m=X.shape[0]
    blocknum=int(m/batchsize)
    index=int(itr%blocknum)
    res_x=X[index*batchsize:(index+1)*batchsize,:]
    res_y=y[index*batchsize:(index+1)*batchsize,:]
    return res_x,res_y #NP


# In[48]:


def y_trans(y):
    y_t=np.zeros((len(y),k))
    for i in range(len(y)):
        y_t[i,int(y[i])]=1
    return y_t


# In[49]:


def get_gd(weight,X,y,learning_rate):
    m=X.shape[0]
    l1=y_trans(y)
    l2=softmax(np.dot(X,weight))
    gradient=learning_rate*np.dot(X.T,l2-l1)
    return gradient


# In[50]:


def mnist_train(X,y,learning_rate):
    [m,n]=X.shape
    w=np.random.random((n,k))
    for i in range(1000):
        train_x,train_y=batch_data(X,y,i)
        grad=get_gd(w,train_x,train_y,learning_rate)
        w=w-grad
    return w


# In[51]:


def mnist_predict(weight,X):
    y_pred=np.argmax(softmax(np.dot(X,weight)), axis=1)
    return y_pred #NP


# In[52]:


def err(y_pred,y):
    return np.count_nonzero(y_pred-y)/len(y) #faster than iteration 
#NP


# In[53]:


def confusionmatrix(y1,y2): #y1 is actual,y2 is predict
    cm=np.zeros((k,k))
    for i in range(len(y2)):
        cm[int(y1[i]),int(y2[i])]+=1
    return cm


# In[54]:


weight=mnist_train(train_x,train_y,0.05)
y_pred=mnist_predict(weight,test_x).reshape(-1,1)
np.savetxt('q6_weight',weight)
print(1-err(y_pred,test_y))


# In[55]:


cm=confusionmatrix(y_pred,test_y)
np.set_printoptions(suppress=True)
print (cm)


# In[ ]:




