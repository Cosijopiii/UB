import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.spatial import distance
import math
import PS

data_size = 260
step_size = 1

train_data = pd.read_csv('datasets/heart-disease-uci/heart.csv')

train_data = train_data[0:data_size]

labels = train_data.columns[:-1]

X = train_data[labels]
y = train_data['target']

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

def train_ub(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plot_2d_space(X_pca, y, 'Imbalanced dataset (2 PCA components)')
    model = svm.SVC(gamma='scale')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def undersampling(X, y, items):

    X_u = X.drop(X.index[items])
    y_u = y.drop(y.index[items])

    return X_u, y_u

def nearest(p, q):
    d = float("inf")
    n_item = None
    for i in range(len(q)):
        item = q.iloc[i]
        c_d = distance.euclidean(item, p)
        if d > c_d:
            d = c_d
            n_item = item
    return n_item

def average(p, q):
    new_item = p.copy(deep = True)
    for i, v, w in zip(range(len(p)), p, q):
        new_item[i] = float(int((v + w) / 2))
    return new_item

# Function to insert row in the dataframe 
def insert_row(row_number, df, row_value): 
    # Starting value of upper half 
    start_upper = 0

    # End value of upper half 
    end_upper = row_number 

    # Start value of lower half 
    start_lower = row_number 

    # End value of lower half 
    end_lower = df.shape[0] 

    # Create a list of upper_half index 
    upper_half = [*range(start_upper, end_upper, 1)] 

    # Create a list of lower_half index 
    lower_half = [*range(start_lower, end_lower, 1)] 

    # Increment the value of lower half by 1 
    lower_half = [x.__add__(1) for x in lower_half] 

    # Combine the two lists 
    index_ = upper_half + lower_half 

    # Update the index of the dataframe 
    df.index = index_ 

    # Insert a row at the end 
    df.loc[row_number] = row_value 
    
    # Sort the index labels 
    df = df.sort_index() 

    # return the dataframe 
    return df 

def oversampling(X, y, items):

    X_o = X.copy(deep = True)
    y_o = y.copy(deep = True)

    for item in items:
        n_item = nearest(X.iloc[item], X)
        new_item = average(X.iloc[item], n_item)
        X_o = insert_row(item+1, X_o, new_item)
        y_o = insert_row(item+1, y_o, y_o[item])

    return (X_o, y_o)

lam = 2
F_s=[]
S_D=[]
SOL_D=[]
epsilon=1
st = "under"

for k in range(int(data_size)):


    X_c = None
    y_c = None

    a_min = y.loc[y == 0].index[0]
    a_max = y.loc[y == 0].index[-1]
    b_min = y.loc[y == 1].index[0]
    b_max = y.loc[y == 1].index[-1]

    u_items = None
    o_items = None

    a_len = a_max - a_min
    b_len = b_max - b_min

    maj_max = a_max if a_len > b_len else b_max
    maj_min = a_min if a_len > b_len else b_min

    min_max = b_max if a_len > b_len else a_max
    min_min = b_min if a_len > b_len else a_min

    if st == "under":
        if k * step_size < (maj_max - maj_min):
            u_items = np.random.choice(np.arange(maj_min, maj_max), k * step_size, replace=False)
        else:
            break
    else:
        if k * step_size < (min_max - min_min):
            o_items = np.random.choice(np.arange(min_min, min_max), k * step_size, replace=False)
        else:
            break

    if st == "under":
        X_c, y_c = undersampling(X, y, u_items)
    else:
        X_c, y_c = oversampling(X, y, o_items)

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

        if R < 0.01:
            SOL_D.append((S_D[k-1], F_s[k-1]))

#for s in sorted(SOL_D, key=lambda x: x[1]):
plotarr=[]
for s in SOL_D:
    print(s[1])
    plotarr.append(s[1])
    print(s[0][1].value_counts())

plt.plot(plotarr) # plotting by columns
plt.show()