import csv
import math
import matplotlib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt

def irisDataSet():
    """
    Attribute information
     1. sepal length (cm)
     2. sepal width (cm)
     3. petal length (cm)
     4. petal width (cm)
     5. class: Iris-setosa, Iris-versicolor, Iris-virginica
    """
    traincsv = "bezdekIris.data"
    clsToInt = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    with open(traincsv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        target = []
        for row in reader:
            if len(row) < 4:
                break
            r = [float(x) for x in row[0:4]]
            data.append(r)
            target.append(clsToInt[row[4]])
        return (np.array(data), np.array(target))

def forestFiresDataSet(binning):
    """
    Attribute information
     1. X
     2. Y
     3. month
     4. day of the week
     5. FFMC 18.7 to 96.20
     6. DMC 1.1 to 291.3
     7. DC 7.9 to 860.6
     8. ISI 0.0 to 56.10
     9. temperatue in Celsius 2.2 to 33.30
     10. relative humidity in % 15.0 to 100
     11. wind speed in km/h 0.40 to 9.40
     12. rain mm/m2 0.0 to 6.4
     13. burned area of the forest in ha 0.00 to 1090.84
    """
    traincsv = "forestfires.csv"
    with open(traincsv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip first line
        data = []
        target = []
        for row in reader:
            if len(row) < 4:
                break
            r = [float(x) for x in row[4:12]]
            data.append(r)
            area = float(row[12])
            if area == 0:
                dmg = 0
            elif area < 1:
                dmg = 1
            elif area < 10:
                dmg = 2
            elif area < 100:
                dmg = 3
            elif area < 1000:
                dmg = 4
            else:
                dmg = 5
            if binning:
                target.append(dmg) 
            else:
                target.append(math.log(area + 1))
        return (np.array(data), np.array(target))

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
    data, target = forestFiresDataSet(False)

    # split train and test data
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size = 0.3)

    # pca
    pca = PCA(n_components=8)
    pca.fit(train_x)
    train_pca_x = pca.transform(train_x)

    # regressor
    clf = DecisionTreeRegressor()
    clf = clf.fit(train_pca_x, train_y)

    # predict
    test_pca_x = pca.transform(test_x)
    test_y_predict = clf.predict(test_pca_x)
    print (test_y_predict)
    print (test_y)

    # performance
    accuracy = metrics.accuracy_score(test_y, test_y_predict)
    return (accuracy)

def tree2pic():
    # load data
    data, target = forestFiresDataSet(False)

    # split train and test data
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size = 0.3)

    # pca
    pca = PCA(n_components=8)
    pca.fit(train_x)
    train_pca_x = pca.transform(train_x)

    # regressor
    clf = DecisionTreeRegressor()
    clf = clf.fit(train_pca_x, train_y)

    # predict
    test_pca_x = pca.transform(test_x)
    test_y_predict = clf.predict(test_pca_x)

    # draw
    plt.scatter(test_y_predict, test_y)
    plt.show()

# test 10 times and average them
def avg10of(model):
    sum = 0
    for i in range(10):
        sum += model()
    print (sum / 10)

avg10of(tree1)
# avg10of(tree2)
tree2pic()
