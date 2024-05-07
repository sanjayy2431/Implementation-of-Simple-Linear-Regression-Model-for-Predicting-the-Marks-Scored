# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step1: Start the program     
Step2: Import the standard Libraries.    
Step3: Set variables for assigning dataset values.    
Step4: Import linear regression from sklearn.     
Step5: Assign the points for representing in the graph    
Step6: Predict the regression for marks by using the representation of the graph.     
Step7: Compare the graphs and hence we obtained the linear regression for the given datas.    
Step8: Stop the program   





## Program:                
Program to implement the simple linear regression model for predicting the marks scored.    
Developed by: v.sanjay    
RegisterNumber: 212223230188                                                                                                                                        
                                                                                                                                                                    
                                                                                                                                                                    
             
from sklearn.linear_model import LinearRegressionimport pandas as pd    
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
                                                                                                                                                                          

                                                                                                                                                                  
                                                                                                                                                                  
                                                                                                                                                                  
                                                                                                                                                                           
## Output:                                                                                                                                                          
                             
![Screenshot 2024-02-23 100336](https://github.com/sanjayy2431/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365143/e5c7310b-d3a0-48e3-9ebe-ff81fb1ff2da)
## TRAINING DATASET GRAPH:
 ![image](https://github.com/sanjayy2431/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365143/de48431c-ba38-4006-a613-356e01cfc407)
## TEST DATASET GRAPH: 
 ![image](https://github.com/sanjayy2431/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365143/1043dca9-cb56-4162-b0b6-19d1ee34ef88)
## VALUES OF MSE,MAE,RMSE:
![image](https://github.com/sanjayy2431/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365143/f9e789d0-8dec-4253-80e0-3a2d74d458ee)
## Result:
 Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
