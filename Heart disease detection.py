# -*- coding: utf-8 -*-
"""
@ Predicting Whether patient has Heart disease or not

Created on Wed Jul 20 20:11:37 2022

@author: Vivek Kute
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("D:\Machine Learning\dataset_files/heart.csv")
X = data.iloc[:, : -1].values
Y = data.iloc[:, 13].values
Y = np.reshape(Y, (1025, 1))

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)
reg2 = LinearRegression()
reg2.fit(X_poly, Y)

print(reg.predict([[43,0,0,132,341,1,0,136,1,3,1,0,3]]))
print(reg2.predict([[43,0,0,132,341,1,0,136,1,3,1,0,3]]))

