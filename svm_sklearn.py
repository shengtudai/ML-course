import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 读取数据特征
def readData(filename):
    datasets = pd.read_csv(filename).T[2:-1]
    datasets = datasets[[i%2==0 for i in range(len(datasets.index))]]
    datasets.index = map(int,datasets.index)
    datasets = np.array(datasets.sort_index())
    return datasets

# 读取数据标签
def readLabel(filename):
    labels = np.array(pd.read_csv(filename))[:,-1]
    return labels

#计算准确率
def accuracy(real,predict):
    num = 0
    for i in range(len(real)):
        if real[i]==predict[i]:
            num +=1
    return num/len(real)
data = readData('datasets.csv')
Y = readLabel('actual.csv')
sc = StandardScaler()
X = sc.fit_transform(data)
pca = PCA(n_components=72)
X = pca.fit_transform(X)
print('数据维度:',X.shape)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.45,random_state=3)
classifier = SVC(kernel='linear')
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
A=B=C=D=0
for i in range(len(Y_test)):
    if Y_test[i]=='AML':
        if Y_pred[i] == 'AML':
            A+=1
        if Y_pred[i]=='ALL':
            B+=1
    if Y_test[i]=='ALL':
        if Y_pred[i] == 'AML':
            C+=1
        if Y_pred[i]=='ALL':
            D+=1
print('TP:',A)
print('FP:',B)
print('FN:',C)
print('TN:',D)
print('准确率：',accuracy(Y_test,classifier.predict(X_test)))
print('支持向量个数：',len(classifier.support_vectors_))