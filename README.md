# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: YUVASHREE R
RegisterNumber:  212224040378
*/
```

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output:
![linear regression using gradient descent](sam.png)

Data Information:
<img width="720" height="147" alt="1" src="https://github.com/user-attachments/assets/8cb5e04f-196b-43ae-b343-6b59b81de4d1" />

Value of X:

<img width="350" height="586" alt="Screenshot 2025-09-01 092717" src="https://github.com/user-attachments/assets/9c53eea4-f12d-4e73-bb10-1991d0c05948" />

Value of X1_Scaled:

<img width="552" height="760" alt="Screenshot 2025-09-01 092727" src="https://github.com/user-attachments/assets/0756c503-33df-4caa-9f7e-d9a7fe4b6332" />


Predicted value:

<img width="489" height="78" alt="Screenshot 2025-09-01 092733" src="https://github.com/user-attachments/assets/a5ed25a7-ee6a-4854-a7ac-8ea75450462b" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
