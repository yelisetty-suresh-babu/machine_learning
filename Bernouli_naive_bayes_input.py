import BernouliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(
        word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


data = pd.read_csv('/Users/yelisettysureshbabu/Desktop/spam.csv', encoding="ISO-8859-1")
data = pd.DataFrame(data)
data = data[['v1', 'v2']].copy()
data.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

data.label[data.label == 'ham'] = 0
data.label[data.label == 'spam'] = 1
data['text'] = data['text'].apply(stemming)
x = data.drop(columns=['label'], axis=1)
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=3, stratify=y)

bernouli = BernouliNB()


def get_score(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


print(get_score(bernouli, x_train,  y_train, x_test, y_test))
