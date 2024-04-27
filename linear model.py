# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:20:46 2024

@author: abc
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv('homeprices.csv')
X=data[['area']]
y=data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mol=LinearRegression()
mol.fit(X_train, y_train)
y_pred=mol.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
plt.scatter(X_train,y_train)
plt.plot(X_train,mol.predict(X_train))