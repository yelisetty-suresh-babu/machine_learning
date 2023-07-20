import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def score1(y,y_):
  c=0
  for i in range(len(y)):
    if(y[i]==y_[i]):
      c+=1
  return (c/len(y))*100



class BernouliNB:
  def __init__(self):
    self.n=len(self.feature_names_spam.shape)+len(self.feature_names_ham.shape)
  def _prior(self):
    self.ham_prob=self.ham_mail/(self.ham_mail+self.spam_mail)
    self.spam_prob=self.spam_mail/(self.ham_mail+self.spam_mail)
  def feature_extract(self,x,y):
    data_train=pd.concat([x, y], axis=1, join='inner')
    data_spam=data_train[data_train['label']==1]
    data_ham=data_train[data_train['label']==0]
    self.spam_mail,_=data_spam.shape
    self.ham_mail,_=data_ham.shape
    cv=CountVectorizer()
    data_ham_cv=cv.fit_transform(data_ham['text'].values)
    self.feature_names_ham=cv.get_feature_names_out()
    data_spam_cv=cv.fit_transform(data_spam['text'].values)
    self.feature_names_spam=cv.get_feature_names_out()

  def probability_calc(self,text):
    text=text.split()
    ham_prob=self.ham_prob
    for word in text:
      num=1
      if word in self.feature_names_ham:
        num+=1
      den=self.n+2
      ham_prob*=(num/den)
    spam_prob=self.spam_prob
    for word in text:
      num=1
      if word in self.feature_names_spam:
        num+=1
      den=self.n+2
      spam_prob*=(num/den)

    if ham_prob>spam_prob:
      return 0
    else:
      return 1
 

  def fit(self,x,y):
    self.x=x
    self.y=y
    self.feature_extract(x,y)
    self._prior()
    y_calc=x['text'].apply(self.probability_calc)
    print(y.to_numpy())
    print(y_calc.to_numpy())
    print(score1(y.to_numpy(),y_calc.to_numpy()))

  def predict(self,x):
    return x['text'].apply(self.probability_calc).to_numpy()
  def score(self,x,y):
    y_calc=self.predict(x)
    return score1(y.to_numpy(),y_calc)




