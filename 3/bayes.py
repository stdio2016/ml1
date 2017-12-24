import math
import matplotlib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

from read import irisDataSet, forestFiresDataSet

def bayes1():
    # load data
    data, target = irisDataSet()

    # split train and test data
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size = 0.3)

    # classifier
    clf = GaussianNB()
    clf = clf.fit(train_x, train_y)

    # predict
    test_y_predict = clf.predict(test_x)

    # performance
    accuracy = metrics.accuracy_score(test_y, test_y_predict)
    return (accuracy)

def bayes2():
    # load data
    data, target = forestFiresDataSet(True)

    # split train and test data
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size = 0.3)

    # classifier
    clf = GaussianNB()
    clf = clf.fit(train_x, train_y)

    # predict
    test_y_predict = clf.predict(test_x)

    # performance
    accuracy = metrics.accuracy_score(test_y, test_y_predict)
    return (accuracy)

# test 10 times and average them
def avg10of(model):
    sum = 0
    for i in range(10):
        sum += model()
    print (sum / 10)

avg10of(bayes1)
avg10of(bayes2)
