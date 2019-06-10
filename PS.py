import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import math
from imblearn.under_sampling import RandomUnderSampler
df_train = pd.read_csv('datasets/heart-disease-uci/heart.csv')
df_train=df_train[0:240]

labels = df_train.columns[:10]

X = df_train[labels]
y = df_train['target']

rus = RandomUnderSampler(return_indices=True)
X_rus, y_rus, id_rus = rus.fit_sample(X, y)

def train_ub(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

lam=2
F_s=[]
S_D=[]
SOL_D=[]
epsilon=1
for k in range(len(id_rus/5)):
    b=k*5
    c=5*(k+1)
    for t in range(b,c):
        X.drop(X.index[id_rus[t]])
        y.drop(y.index[id_rus[t]])
    acc=train_ub(X,y)
    F_s.append(acc)
    S_D.append([X,y])
    if len(F_s)-lam > 0:
        sum_temp=0
        for a in range(k-lam,k):
                    d=F_s[a]-F_s[a+1]
                    d=d*d
                    sum_temp=sum_temp+d
        R=math.sqrt(sum_temp)
        if R < epsilon:
            SOL_D.append([S_D[k-1],F_s[k-1]])

