import numpy as np
import pandas as pd
from altair import Chart
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_moons
from pandas import DataFrame
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import math
from imblearn.under_sampling import RandomUnderSampler
# df_train = pd.read_csv('datasets/heart-disease-uci/heart.csv')
# df_train=df_train[0:260]
#
# labels = df_train.columns[:10]
#
# X = df_train[labels]
# y = df_train['target']
#
# rus = RandomUnderSampler(return_indices=True,sampling_strategy='majority')
# X_rus, y_rus,id_rus = rus.fit_sample(X, y)
# #id_rus=rus.return_indices
# def train_ub(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#     model = svm.SVC(gamma='scale')
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#
#     print('cc')
#     return accuracy
#
# lam=2
# F_s=[]
# S_D=[]
# SOL_D=[]
# epsilon=1
#
# for k in range(int((len(id_rus)/5)-1)):
#
#     c=5*(k+1)
#     #for t in range(0,c):
#     X_c=X.drop(X.index[id_rus[0:c]])
#     y_c=y.drop(y.index[id_rus[0:c]])
#     print(y_c.value_counts())
#     acc=train_ub(X_c,y_c)
#
#     F_s.append(acc)
#
#     S_D.append([X_c,y_c])
#     if len(F_s)-lam > 0:
#         sum_temp=0
#         for a in range(k-lam,k):
#                     d=F_s[a]-F_s[a+1]
#                     d=d*d
#                     sum_temp=sum_temp+d
#         R=math.sqrt(sum_temp)
#         if R < epsilon:
#             SOL_D.append([S_D[k-1],F_s[k-1]])
# #for s in SOL_D:
#     #print(s[1])
#     #print(s[0][1].value_counts())
import random
import math
def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
def randCir(a,b,c,N,rs):
    rho = np.sqrt(np.random.uniform(a, b, N))
    phi = np.random.uniform(0, 2*np.pi, N)

    x = rho * np.cos(phi)-rs
    y = rho * np.sin(phi)-rs
    if c==1:
        Y=np.ones(N)
    else:
        Y=np.zeros(N)
    X=[x,y]
    # plt.scatter(x, y, s = 4)
    # plt.show()
    return X,Y


def genData(a,b,c1,c2):
    N1=a
    N2=b
    X1, Y1 = randCir(0, 1, 1, int(N1/2),c1)
    X2, Y2 = randCir(0, 1, 0, int(N2/2),c2)

    X3, Y3 = randCir(0, 2, 1, int(N1/2), c1 )
    X4, Y4 = randCir(0, 2, 0, int(N2/2), c2 )

    d1=[]

    for i in range(0,int(N1/2)):
        x = X1[0][i]
        y = X1[1][i]
        d1.append([x, y, Y1[i]])
        x = X3[0][i]
        y = X3[1][i]
        d1.append([x, y, Y1[i]])
    for i in range(0,int(N2/2)):
        x = X2[0][i]
        y = X2[1][i]
        d1.append([x, y, Y2[i]])
        x = X4[0][i]
        y = X4[1][i]
        d1.append([x, y, Y2[i]])
    df = pd.DataFrame(d1, columns = ['x', 'y','c'])
    return df
#
# df=genData(40,300,.5,0)
# d=df.columns[:2]
# co=df.columns[2]
# X = df[d]
# y = df[co]
#
# groups = df.groupby('c')
# fig, ax = plt.subplots()
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)
# ax.legend()
#
# plt.show()

# df.plot(kind='scatter',x='x',y='y',c='c',color='red')
# plt.show(
def makeblob(a,b):
    # generate 2d classification dataset
    X, y = make_moons(n_samples=a, noise=b)
    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()
    return df
