# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:30:21 2018

@author: Dell
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir("C:\\pydata\\")

mut=pd.read_csv("New mutual.csv")
mut
bsemonthlyret1 = mut['NSE'].pct_change()
bseclean_monthly_returns = bsemonthlyret1.dropna(axis=0)

icicimonthlyret1= mut['Ret_Sch'].pct_change()
iciciclean_monthly_returns = icicimonthlyret1.dropna(axis=0)
bsemonthlyret1
icicimonthlyret1

X=mut[['Expense_R1','Loadfee1','Aseet_Size1']]
y=mut['Ret_Sch']
#X=X.reshape(-1,1)
X
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=4) 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
print(reg.coef_)
print(reg.intercept_)


##.........................SHARPE RATIO.....................................
import statistics
mean1=iciciclean_monthly_returns.mean()
arr=np.array(mut['Ret_Sch'])
std1=statistics.stdev(arr)
return_free_annual=0.065
return_daily=((1+0.065)**(1/360))-1
sharpe_A_daily=(mean1-return_daily)/std1
sharpe_A_annual=sharpe_A_daily*(252**(0.5))


#,,,,,,,,,,,,,,,,,,,,,,,,SECOND SCHEME,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
icicimonthlyret2= mut['An_Sch'].pct_change()
iciciclean_monthly_returns2 = icicimonthlyret2.dropna(axis=0)
icicimonthlyret2

X1=mut[['Expense_A1','Loadfee','Asset_Size2']]
y1=mut['An_Sch']
#X=X.reshape(-1,1)
X1
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size=0.20,random_state=4) 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

y_pred1 = reg.predict(X_test)
print(reg.coef_)
print(reg.intercept_)


#.................................SHARPE2..........................................................
import statistics
mean2=iciciclean_monthly_returns2.mean()
arr2=np.array(mut['An_Sch'])
std2=statistics.stdev(arr2)
return_free_annual2=0.065
return_daily2=((1+0.065)**(1/360))-1
sharpe_A_daily2=(mean2-return_daily2)/std2
sharpe_A_annual2=sharpe_A_daily2*(252**(0.5))
