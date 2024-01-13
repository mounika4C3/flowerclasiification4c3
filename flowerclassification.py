# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
      
"""
import pandas as pd
#import numpy as np

data=pd.read_csv(r"C:\Users\yaman\OneDrive\Desktop\iris.csv")

data.shape #no.of rows and columns
data.size  #total no.of elements
data.head()
data.info()
data.describe()   #to find range of values in each column

##splitting data into dependent and independent variables

#iloc=integer location
## values------used to convert dataframe into matrix.  since algorithm only accepts matrix format

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

##splitting data for training and testing purpose
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=7)  #test_size denotes 20% of data splitted for testing and remaining for training


##importing kNN algorithm named KNeighborsClassifier from neighbors module
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest) #prediction values

#comparing prediction values given by algorithm and testing values and checking for accuracy

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100) 

## giving input and checking output
print(model.predict([[7.3,5.5,4.3,1.9]]))