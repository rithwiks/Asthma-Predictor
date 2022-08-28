import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import exp
plt.rcParams["figure.figsize"] = (20, 5)

dataF = pd.read_csv("/Users/rithwikseth/Documents/Code/Logisitc Regression/aF.csv",  header=None)
dataF.head()
dxf = np.array(dataF[0])
print(dxf)
dyf = np.array(dataF[1])
dataS = pd.read_csv("/Users/rithwikseth/Documents/Code/Logisitc Regression/aS.csv",  header=None)
dataS.head()
dxs = np.array(dataS[0])
print(dxs)
dys = np.array(dataS[1])
#dx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#dy = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
dx = np.append(dxf, dxs)
dy = np.append(dyf, dys)
plt.scatter(dx, dy)
plt.show()

'''X_train1, X_test1, y_train1, y_test1 = train_test_split(dxf, dyf, test_size=0.30)
X_train2, X_test2, y_train2, y_test2 = train_test_split(dxs, dys, test_size=0.30)
X_train = np.append(X_train1, X_train2)
X_test = np.append(X_test1, X_test2)
y_train = np.append(y_train1, y_train2)
y_test = np.append(y_test1, y_test2)'''
X_train = dx
X_test = dx
y_train = dy
y_test = dy
def normalize(X):
    return X - X.mean()

# Method to make predictions
def predict(X, b0, b1):
    return np.array([1 / (1 + exp(-1*b0 + -1*b1*x)) for x in X])

# Method to train the model
def logistic_regression(X, Y):

    X = normalize(X)

    # Initializing variables
    b0 = 0
    b1 = 0
    L = 0.001
    epochs = 2000

    for epoch in range(epochs):
        y_pred = predict(X, b0, b1)
        D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b0
        D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1
        # Update b0 and b1
        b0 = b0 - L * D_b0
        b1 = b1 - L * D_b1
    
    return b0, b1, D_b0, D_b1
b0, b1, db0, db1 = logistic_regression(X_train, y_train)


X_test_norm = normalize(X_test)
y_pred = predict(X_test_norm, b0, b1)
print(y_pred)
print(b0, b1)
print(db0, db1)
y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

'''plt.clf()
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred, c="red")
plt.show()'''
accuracy = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        accuracy += 1
print(f"Accuracy = {accuracy / len(y_pred)}")