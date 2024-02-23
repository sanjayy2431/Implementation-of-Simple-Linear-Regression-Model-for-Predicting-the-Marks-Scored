# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: v.sanjay
RegisterNumber: 212223230188  
*/
```

## Output:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```
![Screenshot 2024-02-23 100336](https://github.com/sanjayy2431/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365143/e5c7310b-d3a0-48e3-9ebe-ff81fb1ff2da)

![image](https://github.com/sanjayy2431/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365143/de48431c-ba38-4006-a613-356e01cfc407)

![image](https://github.com/sanjayy2431/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365143/1043dca9-cb56-4162-b0b6-19d1ee34ef88)

![image](https://github.com/sanjayy2431/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365143/f9e789d0-8dec-4253-80e0-3a2d74d458ee)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
