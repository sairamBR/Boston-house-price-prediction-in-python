# Boston-house-price-prediction-in-python
This is an machine learning based project to predicted the house price in Boston with supervised learning technique using Gradient Descent Algorithm.  
# Steps involed in this process making predictive machine learning model
   *Aquring the Dataset
   *Data preprocessing
   *spliting of dataset
   *choosing the suitable algorithm for your model(here we are using the Gradient Descent Algorithm)
   *train your model with different correlation value
# Begin you have be comfortable with anaconda and Jupyter notebook
  to known and getting started with anaconda and jupyter notebook vist the links given below
  https://youtu.be/1aBuUSSg0zw
  subscribe my channel for more videos eductional videos
  get to known more about the Anaconda and jupyter notebook vist the learning section in the anaconda navigator
# open the jupyter notebook from Anaconda navigator
  after the anaconda navigator import commanly used libarires for machine learning as i did in the code
  
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

  my suggestion to you people just keep my code as the reference as do it by your own then only it make sense
After import liberires import the dataset. the Boston Dataset is already available in Scikit library so that you can directly do it from it

After doing this read the desceritpion  of the dataset before proceeding future.

boston = load_boston()
print (boston.DESCR)

You can include this data by using the 'MASS' library. The data has following features, medv being the target (dependent) variable:

crim - per capita crime rate by town
zn - proportion of residential land zoned for lots over 25,000 sq.ft
indus - proportion of non-retail business acres per town
chas - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
nox - nitric oxides concentration (parts per 10 million)
rm - average number of rooms per dwelling
age - proportion of owner-occupied units built prior to 1940
dis - weighted distances to five Boston employment centres
rad - index of accessibility to radial highways
tax - full-value property-tax rate per USD 10,000
ptratio - pupil-teacher ratio by town
black - proportion of blacks by town
lstat - percentage of lower status of the population
medv - median value of owner-occupied homes in USD 1000’s
# Dataframing 
  these lines are used to Frame the dataset
  
  features = pd.DataFrame(boston.data, columns=boston.feature_names)
  features
 framing the traget values
  target = pd.DataFrame(boston.target,columns=['target'])
  target
 let find the maxmium and miminim values of target
  max(target['target'])
  min(target['target'])
 the concantanate is used to frame the attirbutes and target data into single dataframe
  df =pd.concat([features,target],axis=1)
  df
 let begin with finding the correlation values
  corr = df.corr('pearson')
 To take the absolute value of the correlation
  corrs = [abs(corr[attr]['target'])for attr in list(features)]
  
 let plot the graph with highest correlation value using matplotlib
   #Make a list of pairs [(corr, feature)]
l = list(zip(corrs, list(features)))

#sort the list of pairs in reverse/dcesending order
#with the correlation value as the key for sorting
l.sort(key = lambda x : x[0], reverse=True)

#"unzip" pairs to two lists
#zip(*l) - takes a list that looks like 
corrs,labels = list(zip((*l)))

#plot correlations with respect to the variable as a bar graph
index = np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index, corrs, width=0.5)
plt.xlabel('attributes')
plt.ylabel('correlation with the target variable')
plt.xticks(index, labels)
plt.show()

 
