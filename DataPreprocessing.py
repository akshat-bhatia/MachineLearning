# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Data.csv')

#Dividing the dataset based on dependent and independent decision variable

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='Nan', strategy='mean', axis=0)
imputer = Imputer.fit(X)
X[:, 1:3] = Imputer.transform(X[:, 1:3])

#Encoding catregorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
