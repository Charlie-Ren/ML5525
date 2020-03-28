#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np, pandas as pd
from matplotlib import pyplot as pl
feat=pd.read_csv("IRISFeat.csv",header=None)
label=pd.read_csv("IRISlabel.csv",header=None)
idx=np.random.permutation(feat.index)# shuffle
X_shuffle=feat.reindex(idx).to_numpy()
y_shuffle=label.reindex(idx).to_numpy()
X_shuffle=np.c_[X_shuffle,np.ones(len(X_shuffle))] #add one extra column with only 1 exists. 
train_err=[-1,-1,-1,-1,-1]
valid_err=[-1,-1,-1,-1,-1]
#data process


# In[40]:


def sigmoid(para):
    b=np.negative(para)
    return (1.0/(1+np.exp(b))) #it works without changing what it is supposed to be


# In[41]:


def y_predict_class(X_valid,model_weights,model_intercept):
    weight=model_weights # We include intercept in model_weights, so we can do it directly by np.dot.
    y_pred=sigmoid(np.dot(X_valid,weight)) #根据sigmoid判断
    y_pred[y_pred>=1/2]=1
    y_pred[y_pred<1/2]=0
    return y_pred


# In[42]:


def cost(y_pred,y_train):
    count=0;
    for i in range(len(y_pred)):
        if (y_pred[i]!=y_train[i]):
            count+=1
    return count/len(y_pred)


# In[46]:


def train(X_train,y_train):
    weights=np.random.random((3,1))  # get the intercept into the weights so we can do multipy directly.
    for i in range(500):
        y_pred=y_predict_class(X_train,weights,weights[2]) #get 2 rows
        err=cost(y_pred,y_train)
        if err<0.001:
            break
        lr=0.005
        hout=sigmoid(np.dot(X_train,weights)) #pred-ground_truth
        gradient=np.dot(np.transpose(X_train),(hout-y_train))
        weights=weights-lr*gradient
    #print(err)
    return weights,weights[2]


# In[47]:


def get_next_train_valid(X_shuffle,y_shuffle,itr):
    if itr==0:
        X_valid=X_shuffle[:30]
        y_valid=y_shuffle[:30]
        X_train=X_shuffle[30:]
        y_train=y_shuffle[30:]
    elif itr==1:
        X_valid=X_shuffle[30:60]
        y_valid=y_shuffle[30:60]
        X_train=np.delete(X_shuffle,range(30,60),0)
        y_train=np.delete(y_shuffle,range(30,60),0)
    elif itr==2:
        X_valid=X_shuffle[60:90]
        y_valid=y_shuffle[60:90]
        X_train=np.delete(X_shuffle,range(60,90),0)
        y_train=np.delete(y_shuffle,range(60,90),0)
    elif itr==3:
        X_valid=X_shuffle[90:120]
        y_valid=y_shuffle[90:120]
        X_train=np.delete(X_shuffle,range(90,120),0)
        y_train=np.delete(y_shuffle,range(90,120),0)
    elif itr==4:
        X_valid=X_shuffle[120:]
        y_valid=y_shuffle[120:]
        X_train=X_shuffle[:120]
        y_train=y_shuffle[:120]
    return X_train,y_train,X_valid,y_valid


# In[66]:


for i in range(5):
    X_train,y_train,X_valid,y_valid=get_next_train_valid(X_shuffle,y_shuffle,i)
    weight,intercept=train(X_train,y_train)
    y_pred0=y_predict_class(X_train,weight,intercept)
    train_err[i]=cost(y_pred0,y_train)
    y_pred1=y_predict_class(X_valid,weight,intercept)
    #pos1,neg1=count(y_valid) #for confusion matrix
    #pos2,neg2=count(y_pred1)
    #print(",valid,",pos1,neg1)
    #print(",pred",pos2,neg2)
    valid_err[i]=cost(y_pred1,y_valid)
iteration=[1,2,3,4,5]
pl.plot(iteration,train_err,label="training error")
pl.plot(iteration,valid_err,label="validation error")
pl.xlabel("iteration time")
pl.ylabel("error rate")
pl.grid(True)
pl.legend()
pl.show()


# In[63]:


def count(y_train):
    pos=0
    neg=0
    for i in range(len(y_train)):
        if y_train[i]==1:
            pos+=1
        else:
            neg+=1
    return pos,neg


# In[ ]:




