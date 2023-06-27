import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




class logistic_regression:
    def __init__(self,lr=0.01,num_itr=1000) :
        self.lr=lr
        self.num_itr=num_itr
        self.weights=None
        self.bias=None
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.num_itr):
            # y_pred=1/(1+np.exp(-np.dot(x.T,self.weights)+self.bias))
            linear=np.dot(x,self.weights)+self.bias
            prob=1/(1+np.exp(-linear))

            dw=(1/n_samples)*np.dot(x.T,(prob-y))
            db=(1/n_samples)*np.sum(prob-y)

            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db



    def predict(self,x):
        temp=1/(1+np.exp(-(np.dot(x,self.weights)+self.bias)))
        return [0 if i<0.5 else 1 for i in temp]
    




bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = logistic_regression(lr=0.01)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)
    


