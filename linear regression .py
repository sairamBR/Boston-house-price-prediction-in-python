#!/usr/bin/env python
# coding: utf-8

# In[38]:


#import the commonly used machine learning libraries and libraries required for this project
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML


# In[39]:


#to show the descripiton of dataset
boston = load_boston()
print (boston.DESCR)


# In[40]:


# to put the data into pandas data framework
features = pd.DataFrame(boston.data, columns=boston.feature_names)
features


# In[41]:


#to show only specific feature of the dataset
features['AGE']


# In[42]:


target = pd.DataFrame(boston.target,columns=['target'])
target


# In[43]:


max(target['target'])


# In[44]:


min(target['target'])


# In[45]:


#concatenate features and target into single dataframe
# aixs = 1 caoncat the dataframe in column wise
df =pd.concat([features,target],axis=1)
df


# In[46]:


#use the round(decimals=2)to set the presicion of 2 decimals

df.describe().round(decimals=2)


# In[47]:


# to collect the correlation value between the every column of the data
corr = df.corr('pearson')

# To take the absolute value of the correlation
corrs = [abs(corr[attr]['target'])for attr in list(features)]

#Make a list of pairs [(corr, feature)]
l = list(zip(corrs, list(features)))

#sort the list of pairs in reverse/dcesending order
# with the correlation value as the key for sorting
l.sort(key = lambda x : x[0], reverse=True)

# "unzip" pairs to two lists
# zip(*l) - takes a list that looks like 
corrs,labels = list(zip((*l)))

# plot correlations with respect to the variable as a bar graph
index = np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index, corrs, width=0.5)
plt.xlabel('attributes')
plt.ylabel('correlation with the target variable')
plt.xticks(index, labels)
plt.show()


# In[48]:


X = df['LSTAT'].values
Y = df['target'].values


# In[49]:


#print 5 values of y before normalizing 
print(Y[:5])


# In[50]:


x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X.reshape(-1, 1))
X = X[:, -1]
y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y.reshape(-1, 1))
Y = Y[:, -1]


# In[51]:


print(Y[:5])


# In[52]:


def error(m, x, c, t):
    N = x.size
    e = sum(((m * x + c) - t) ** 2)
    return e * 1/(2 * N)


# In[53]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2)


# In[54]:


def update(m, x, c, t, learning_rate):
    grad_m = sum(2 * ((m * x + c) - t) * x)
    grad_c = sum(2 * ((m * x + c) - t))
    m = m - grad_m * learning_rate
    c = c - grad_c * learning_rate
    return m,c
    


# In[55]:


#implementing descent gradient algorithm
def gradient_descent(init_m, init_c, x, t, learning_rate, iterations, error_threshold):
    m = init_m
    c = init_c
    error_values = list()
    mc_values = list()
    for i in range(iterations):
        e = error(m, x, c, t)
        if e < error_threshold:
            print('error less than the threshold. stopping gradient descent')
            break
        error_values.append(e)
        m, c = update (m, x, c, t, learning_rate)
        mc_values.append((m, c))
    return m, c, error_values, mc_values


# In[56]:


get_ipython().run_cell_magic('time', '', 'init_m = 0.9\ninit_c = 0\nlearning_rate = 0.001\niterations = 250\nerror_threshold = 0.001\n\nm, c, error_values, mc_values = gradient_descent(init_m, init_c, xtrain, ytrain, learning_rate, iterations, error_threshold)')


# In[61]:


plt.scatter(xtrain, ytrain, color='r')
plt.plot(xtrain, (m * xtrain + c), color='b')


# In[62]:


plt.plot(np.arange(len(error_values)), error_values)
plt.ylabel('error')
plt.xlabel('Iterations')


# In[63]:


predicted = (m * xtest) + c


# In[64]:


mean_squared_error(ytest, predicted)


# In[65]:


p = pd.DataFrame(list(zip(xtest, ytest, predicted)), columns=['x', 'target_y', 'predicted_y'])
p.head()


# In[66]:


plt.scatter(xtest, ytest, color='b')
plt.plot(xtest, predicted, color='r')


# In[1]:


predicted = predicted.reshape(-1, 1)
xtest = xtest.reshape(-1, 1)
ytest = ytest.reshape(-1, 1)

xtest_scaled = x_scaler.inverse_transform(xtest)
ytest_scaled = y_scaler.inverse_transform(ytest)
predicted_scaled = y_scaler.inverse_transform(predicted)

xtest_scaled = xtest_scaled[:, -1]
ytest_scaled = ytest_scaled[:, -1]
predicted_scaled =predicted_scaled[:, -1]

p = pd.DataFrame(list(zip(xtest_scaled, ytest_scaled,predicted_scaled)), columns=['x', 'target_y', 'predicted_y'])
p = p.round(decimals = 2)
p.head()


# In[ ]:




