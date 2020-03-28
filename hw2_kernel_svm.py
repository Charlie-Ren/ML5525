#!/usr/bin/env python
# coding: utf-8

# In[1]:


global k
k=10
import numpy as np
import pandas as pd
import cvxopt
from matplotlib import cm
from matplotlib import axes
from matplotlib import pyplot as plt
data_in=pd.read_csv("hw2data.csv",header=None).to_numpy() #import data and transfer to array
len1=int(len(data_in)*0.8)
traindata=data_in[:len1]
testdata=data_in[len1:]


# In[2]:


def rbf(X1,X2,sigma):
    return np.exp(-np.linalg.norm(X1-X2)**2/(2*sigma**2))


# In[3]:


def rbf_svm_train(X,y,c,sigma):#return Alpha
    [m,n]=X.shape
    y=y.reshape(-1,1)*1. #make it float
    Gram=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            Gram[i,j]=rbf(X[i],X[j],sigma) #kernel
    cvxopt.solvers.options['show_progress'] = False
    P=cvxopt.matrix(np.outer(y,y)*Gram)
    q=cvxopt.matrix(-np.ones((m,1)))
    G=cvxopt.matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h=cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * c)))
    ans=cvxopt.solvers.qp(P,q,G,h)
    alpha=np.array(ans['x'])
    return alpha


# In[4]:


def predict(test_X,train_X,train_y,alpha,sigma):
    len1=test_X.shape[0]
    len2=train_X.shape[0]
    Gram=np.zeros((len1,len2))
    for i in range(len1):
        for j in range(len2):
            Gram[i,j]=rbf(test_X[i],train_X[j],sigma)
    label= Gram @ (alpha*train_y)
    return np.sign(label)


# In[5]:


def choose(train_data,c,sigma): #if you use k-folder here it takes more than 5 hours to do that.
    error_train=0
    length=int(train_data.shape[0]*0.8)
    train_x=train_data[:length,:2] 
    train_y=train_data[:length,2].reshape(-1,1)*1.
    valid_x=train_data[length:,:2]
    valid_y=train_data[length:,2].reshape(-1,1)*1.
    alpha=rbf_svm_train(train_x,train_y,c,sigma)
    pred_valid=predict(valid_x,train_x,train_y,alpha,sigma)
    error_valid=err(pred_valid,valid_y) #select c,sigma based on training error. Be aware of overfitting!
    return error_valid


# In[6]:


def k_folder_cv(train_data,test_data,c,sigma): #use chosen parameters to get test error and validation error.
    [m,n]=train_data.shape
    per=int(m/k)
    test_x=test_data[:,:2]
    test_y=test_data[:,2].reshape(-1,1)*1.
    err_valid=[]
    err_test=[]
    for i in range(k):
        start=i*per
        end=(i+1)*per
        valid_x=train_data[start:end,:2]
        valid_y=train_data[start:end,2].reshape(-1,1)*1.
        train_x=np.delete(train_data,range(start,end),0)[:,:2]
        train_y=np.delete(train_data,range(start,end),0)[:,2].reshape(-1,1)*1.
        alpha=rbf_svm_train(train_x,train_y,c,sigma)
        pred_valid=predict(valid_x,train_x,train_y,alpha,sigma)
        pred_test=predict(test_x,train_x,train_y,alpha,sigma)
        err_valid.append(err(pred_valid,valid_y))
        err_test.append(err(pred_test,test_y))
    return err_valid,err_test


# In[7]:


def err(label,test_y):
    count=0
    for i in range(len(test_y)):
        if label[i]!=test_y[i]:
            count+=1
    return count/len(test_y)


# In[8]:


C= [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
sigma= [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
valid1=[]
for i in range(len(C)):
    temp1=[] #for train
    for j in range(len(sigma)):
        acc1=choose(traindata,C[i],sigma[j])
        temp1.append(acc1)
        print("C="+str(C[i]),"sigma="+str(sigma[j]),acc1)
    valid1.append(temp1)


# In[9]:


plt.imshow(valid1)
plt.colorbar()
c1=['0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000']
sigma1=['0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000']
lengtt=len(c1)
plt.yticks(np.arange(lengtt),c1)
plt.ylabel("choice of C")
plt.xticks(np.arange(lengtt),sigma1)
plt.xlabel("choice of sigma")
plt.title("validation error")
plt.show()
np.savetxt("selectparameters",valid1)


# In[10]:


valid,test=k_folder_cv(traindata,testdata,1,0.1) #if c and sigma are both too small, overfitting
plt.plot(valid,'r',label="validation error")
plt.plot(test,'b',label="test error")
plt.xlabel('iteration time')
plt.ylabel('err rate')
plt.title("valiadation and test error")
plt.legend()


# In[ ]:




