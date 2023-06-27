# def diff(a,b):
#     n=0
#     for i in range(len(a)):
#         if(a[i]!=b[i]):
#             n+=1
#     return n


# a=[[0 for j in range(10)] for i in range(10)]


# str=[]
# str.append("1111110")
# str.append("0110000")
# str.append("1101101")
# str.append("1111001")
# str.append("0110011")
# str.append("1011011")
# str.append("0011111")
# str.append("1110000")
# str.append("1111111")
# str.append("1110011")


# for i in range(10):
#     for j in range(10):
#         if( i == j ):
#             a[i][j]=0
#         else:
#             a[i][j]=diff(str[i],str[j])

# t=int(input())
# while(t>0):
#     n=int(input())
#     s=input()

#     c=0
#     for i in range(1,n):
#         c+=diff(str[int(s[i])],str[int(s[i-1])])
#     c+=diff("0000000",str[int(s[0])])
#     print(c)
#     t-=1


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# data= pd.read_csv("/Users/yelisettysureshbabu/Downloads/DSA/machine learning/diabetes.csv")
data = pd.read_csv(f'{os.getcwd()}/diabetes.csv', names=['x', 'y'])


scaler = StandardScaler()

x = data.drop(columns="outcome", axis=1)
y = data['outcome']


scaler.fix()
