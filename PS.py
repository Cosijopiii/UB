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
df_train=df_train[0:260]

labels = df_train.columns[:10]

X = df_train[labels]
y = df_train['target']

rus = RandomUnderSampler(sampling_strategy='not majority')
X_rus, y_rus = rus.fit_sample(X, y)
id_rus=rus.sample_indices_
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

for k in range(int((len(id_rus)/5)-1)):

    c=5*(k+1)
    #for t in range(0,c):
    X_c=X.drop(X.index[id_rus[0:c]])
    y_c=y.drop(y.index[id_rus[0:c]])
    print(y_c.value_counts())
    acc=train_ub(X_c,y_c)
    F_s.append(acc)

    S_D.append([X_c,y_c])
    if len(F_s)-lam > 0:
        sum_temp=0
        for a in range(k-lam,k):
                    d=F_s[a]-F_s[a+1]
                    d=d*d
                    sum_temp=sum_temp+d
        R=math.sqrt(sum_temp)
        if R < epsilon:
            SOL_D.append([S_D[k-1],F_s[k-1]])
#for s in SOL_D:
    #print(s[1])
    #print(s[0][1].value_counts())

