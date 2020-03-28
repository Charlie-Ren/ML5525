#!/usr/bin/env python
# coding: utf-8

# In[1]:


global k
k=10
import numpy as np
import pandas as pd
import cvxopt
from matplotlib import pyplot as plt
data_in=pd.read_csv("hw2data.csv",header=None).to_numpy() #import data and transfer to array
len1=int(len(data_in)*0.8)
traindata=data_in[:len1]
testdata=data_in[len1:]


# In[2]:


def svmfit(X,y,c):
    m,n=X.shape
    y=y.reshape(-1,1)*1. #make it float
    X_mul=y*X
    H=np.dot(X_mul,X_mul.T)*1. #m*m
    #cvxopt
    P=cvxopt.matrix(H)
    q=cvxopt.matrix(-np.ones((m,1)))
    G=cvxopt.matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h=cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * c)))
    cvxopt.solvers.options['show_progress'] = False
   # A=cvxopt.matrix(y.reshape(1,-1)) #shape of y needs to be m*1
   # b=cvxopt.matrix(np.zeros(1))
    ans=cvxopt.solvers.qp(P,q,G,h)
    alpha=np.array(ans['x'])
    w=np.dot((y*alpha).T,X).reshape(-1,1) #get w
    return w


# In[3]:


def label(X,weight):
    y_pred=np.dot(X,weight)
    return np.sign(y_pred)


# In[4]:


def acc(y_pred,y):
    count=0
    for i in range(len(y_pred)):
        if y_pred[i]!=y[i]:
            count+=1
    return 1-(count/len(y_pred))


# In[5]:


def k_folder_cv(train_data,test_data,k,c):
    [m,n]=train_data.shape
    per=int(m/k)
    test_x=test_data[:,:2]
    test_y=test_data[:,2]
    acc_train=0
    acc_valid=0
    acc_test=0
    for i in range(k):
        start=i*per
        end=(i+1)*per
        valid_x=train_data[start:end,:2]
        valid_y=train_data[start:end,2]
        train_x=np.delete(train_data,range(start,end),0)[:,:2]
        train_y=np.delete(train_data,range(start,end),0)[:,2]
        weight=svmfit(train_x,train_y,c)
        y_pred1=label(train_x,weight)
        acc_train+=acc(y_pred1,train_y)
        acc_valid+=acc(label(valid_x,weight),valid_y)
        acc_test+=acc(label(test_x,weight),test_y)
    return acc_train,acc_valid,acc_test


# In[6]:


C= [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
l_train=np.zeros(len(C))
l_valid=np.zeros(len(C))
l_test=np.zeros(len(C))
ind=0
for cc in C:
    x1,x2,x3=k_folder_cv(traindata,testdata,k,cc)
    l_train[ind]=x1/k
    l_valid[ind]=x2/k
    l_test[ind]=x3/k
    ind+=1


# In[7]:


#plot
plt.xlabel("value of C")
plt.ylabel("Acc rate")
plt.plot(np.log10(C),l_train,color='r',label="train acc rate")
plt.plot(np.log10(C),l_valid,color='b',label="valid acc rate")
plt.plot(np.log10(C),l_test,color='g',label="test acc rate") #if we use C instead of log10(C), it will be very bad.
plt.legend()


# In[8]:


#for i in range(len(data_in)):
    #if data_in[i,2]==1:
     #   plt.scatter(data_in[i,0],data_in[i,1],c='r')
    #else:
     #   plt.scatter(data_in[i,0],data_in[i,1],c='b')


# In[9]:


#print (l_train)
#print (l_valid)
#print (l_test)


# In[ ]:




