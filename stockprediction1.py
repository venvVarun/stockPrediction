#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh') 

#load the data
df=pd.read_csv('data/nifty501.csv')
print(df.shape)
#print(df.head(6))
#visualize the close price data
plt.figure(figsize=(16, 8))
plt.title('NIFTY')
plt.xlabel('Days')
plt.ylabel('close price in \u20B9')
plt.plot(df['close'])
plt.show()

#getting close price values
df=df[['close']]
print(df.head(10))

#create data sets to predict x days out into the future
future_days=25

#create anew column (target) shifted 'x' units/days  up
df['prediction']=df[['close']].shift(-future_days)
print(df.tail(10))

 #create the feature set of data and convert it to a numpy array and remove the last n rows/days
n=np.array(df.drop(['prediction'],1))[:-future_days]
print(n)

#creat the target data set(y) and convert it to a numpy array and get all of the target value except for last x rows
y=np.array(df['prediction'])[:-future_days]
print(y)

#splitting the data into 75% training and 25% testing
x_train,x_test,y_train,y_test=train_test_split(n,y, test_size=.25)

#creating the models
#create the decision tree regressor model
tree=DecisionTreeRegressor().fit(x_train, y_train)

#create the linear regression model
lr=LinearRegression().fit(x_train, y_train)

#get the last 'x' rows of the feature data set
x_future=df.drop(['prediction'],1)[:-future_days]
x_future=x_future.tail(future_days)
x_future=np.array(x_future)
print(x_future)

#show the model tree prediction
tree_prediction=tree.predict(x_future)
print(tree_prediction)
print()

#show the model linear regression prediction 
lr_prediction=lr.predict(x_future)
print(lr_prediction)

#visualizing the data
prediction=tree_prediction
valid=df[n.shape[0]:]
valid['prediction']=prediction
plt.figure(figsize=(16,8))
plt.title('model')
plt.xlabel('Days')
plt.ylabel('close price in \u20B9')
plt.plot(df['close'])
plt.plot(valid[['close','prediction']])
plt.legend(['orig','val','pred'])
print(plt.show())
