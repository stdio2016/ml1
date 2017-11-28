import csv
import sys
import numpy
from kdtree import knn, Kdtree, pca, knnClassifier

# only some of the columns in csv are used
# and I need to convert string to number
def selectAttributes(row):
    selected = [int(row[0])] # row id
    for i in range(2, 11):
        selected.append(float(row[i])) # attributes
    selected.append(row[11]) # class
    return selected

def main():
    if (len(sys.argv) < 2):
        print ("Usage: ./validate.py [train data]")
        sys.exit()
    traincsv = sys.argv[1]
    with open(traincsv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip first line
        dat = [selectAttributes(row) for row in reader]
        showValidateResult(dat, 9)
        showPCAValidateResult(dat, 9)

def showValidateResult(data, dimension):
    for k in [1, 5, 10, 100]:
        print ("k = %d" % k)
        accuracy = 0
        i = 0
        for query in data:
            train = data[:i] + data[i+1:]
            tree = Kdtree(train, 9)
            expected = query[-1]
            query = query[1:-1]
            ans = knnClassifier(tree, k, query, dimension)
            if expected == ans:
                accuracy += 1.0
            # print (ans)
            i += 1
        accuracy /= len(data)
        print ("KNN accuracy: %f" % accuracy)

def showValidateResult(data, dimension):
    for k in [1, 5, 10, 100]:
        print ("k = %d" % k)
        accuracy = 0
        i = 0
        for query in data:
            train = data[:i] + data[i+1:]
            tree = Kdtree(train, 9)
            expected = query[-1]
            query = query[1:-1]
            ans = knnClassifier(tree, k, query, dimension)
            if expected == ans:
                accuracy += 1.0
            # print (ans)
            i += 1
        accuracy /= len(data)
        print ("KNN accuracy: %f" % accuracy)

def showPCAValidateResult(data, dimension):
    for k in [1, 5, 10, 100]:
        for useDim in range(2, dimension): # 2 to dimension-1
            print ("k = %d, use dimensions = %d" % (k, useDim))
            accuracy = 0
            i = 0
            for query in data:
                train = data[:i] + data[i+1:]
                mat = [row[1:-1] for row in train]
                E, Q, A, mean, std = pca(mat, useDim)
                for j in range(len(train)):
                    # I cannot modify lists inside train because they come from data
                    train[j] = [train[j][0]] + list(A[j]) + [train[j][-1]]
                tree = Kdtree(train, useDim)

                expected = query[-1]
                query = query[1:-1]
                query = (query - mean) / std
                query = numpy.dot(query, Q)
                ans = knnClassifier(tree, k, query, useDim)
                if expected == ans:
                    accuracy += 1.0
                i += 1
            accuracy /= len(data)
            print ("KNN_PCA accuracy: %f" % accuracy)

main()
