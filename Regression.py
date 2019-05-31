
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:35:46 2019

@author: Akshat
"""

#Simple linear regression
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#Dividing the dataset based on independent variable 
'''X is a matrix'''
X = dataset.iloc[:,:-1].values
#and dependent decision variable
'''Y is a vector'''
Y = dataset.iloc[:,1].values

#Splitting the dataset into training and testing dataset
#Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state = 0 )

#Feature Scaling------ Won't be using here cz the library for linear regression model will handle feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting simple linear regression to the training set
#We will call this object as regressor
 from sklearn.linear_model import LinearRegression
 regressor = LinearRegression()
 regressor.fit(X_train, Y_train)
 
 #Predicting the test set results
 y_pred = regressor.predict(X_test)
 
 #Visualizing the training set results
 plt.scatter(X_train, Y_train, color='red')
 plt.plot(X_train, regressor.predict(X_train), color='black')
 plt.title('Salary vs Experience(Training Set)')
 plt.xlabel('Years of experience')
 plt.ylabel('Salary')
 plt.show()
 
  #Visualizing the test set results
 plt.scatter(X_test, Y_test, color='red')
 plt.plot(X_train, regressor.predict(X_train), color='black')
 plt.title('Salary vs Experience(Test Set)')
 plt.xlabel('Years of experience')
 plt.ylabel('Salary')
 plt.show()
 
 
