import math
import matplotlib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt

from read import irisDataSet, forestFiresDataSet

def tree1():
    # load data
    data, target = irisDataSet()

    # split train and test data
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size = 0.3)

    # pca
    pca = PCA(n_components=2)
    pca.fit(train_x)
    train_pca_x = pca.transform(train_x)

    # classifier
    clf = DecisionTreeClassifier()
    clf = clf.fit(train_pca_x, train_y)

    # predict
    test_pca_x = pca.transform(test_x)
    test_y_predict = clf.predict(test_pca_x)

    # performance
    accuracy = metrics.accuracy_score(test_y, test_y_predict)
    return (accuracy)

def tree2():
    # load data
    data, target = forestFiresDataSet(True)

    # split train and test data
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size = 0.3)

    # pca
    pca = PCA(n_components=8)
    pca.fit(train_x)
    train_pca_x = pca.transform(train_x)

    # classifier
    clf = DecisionTreeClassifier()
    clf = clf.fit(train_pca_x, train_y)

    # predict
    test_pca_x = pca.transform(test_x)
    test_y_predict = clf.predict(test_pca_x)

    # performance
    accuracy = metrics.accuracy_score(test_y, test_y_predict)
    return (accuracy)

# test 10 times and average them
def avg10of(model):
    sum = 0
    for i in range(10):
        sum += model()
    print (sum / 10)

avg10of(tree1)
avg10of(tree2)
