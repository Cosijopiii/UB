import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import math

data_size = 260
step_size = 5

train_data = pd.read_csv('datasets/heart-disease-uci/heart.csv')

train_data = train_data[0:data_size]

labels = train_data.columns[:-1]

X = train_data[labels]
y = train_data['target']

def train_ub(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = svm.SVC(gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

lam = 2
F_s=[]
S_D=[]
SOL_D=[]
epsilon=1

for k in range(int(data_size/5)):


    items = np.random.choice(data_size, k * step_size, replace=False)

    X_c = X.drop(X.index[items])
    y_c = y.drop(y.index[items])

    acc = train_ub(X_c,y_c)

    F_s.append(acc)

    S_D.append([X_c,y_c])
    if len(F_s)-lam > 0:
        sum_temp=0
        for a in range(k-lam,k):
                    d = F_s[a] - F_s[a+1]
                    d = d*d
                    sum_temp = sum_temp+d
        R = math.sqrt(sum_temp)
        if R < epsilon:
            SOL_D.append((S_D[k-1], F_s[k-1]))

for s in sorted(SOL_D, key=lambda x: x[1]):
    print(s[1])
    print(s[0][1].value_counts())
