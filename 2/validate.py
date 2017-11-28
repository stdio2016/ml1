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

main()
