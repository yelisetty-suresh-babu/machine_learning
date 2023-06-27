import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class linear_regression:
    def __init__(self, lr=0.01, num_itr=1000):
        self.lr = lr
        self.num_itr = num_itr
        self.weights = 0
        self.bias = 0

    def fit(self, x, y):
        n_sample,n_feature = x.shape
        self.weights = np.zeros(n_feature)
        self.bias = 0

        for _ in range(self.num_itr):
            y_pred = np.dot(x, self.weights)+self.bias
            dw = (1/n_sample)*np.dot(x.T, (y_pred-y))
            db = (1/n_sample)*np.sum(y_pred-y)

            self.weights = self.weights-self.lr*dw
            self.bias = self.bias-self.lr*db

    def predict(self,x ):
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred


model = linear_regression()

x, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4)
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.2, random_state=2)

model.fit(train_x, train_y)

x_test_pred = model.predict(train_x)

print(x_test_pred)
print(accuracy_score(train_y,x_test_pred ))


