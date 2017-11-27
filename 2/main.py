import csv
import sys
from kdtree import knn, Kdtree, knnClassifier

# only some of the columns in csv are used
# and I need to convert string to number
def selectAttributes(row):
    selected = [int(row[0])] # row id
    for i in range(2, 11):
        selected.append(float(row[i])) # attributes
    selected.append(row[11]) # class
    return selected

def main():
    if (len(sys.argv) < 3):
        print ("Usage: ./run.sh [train data] [test data]")
        sys.exit()
    traincsv = sys.argv[1]
    testcsv = sys.argv[2]
    with open(traincsv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip first line
        dat = [selectAttributes(row) for row in reader]
        tree = Kdtree(dat, 9, 0)
        # showKdtree(Kdtree(dat, 9, 0))
        with open(testcsv, 'r') as csvfile2:
            reader2 = csv.reader(csvfile2)
            next(reader2) # skip first line
            test = [selectAttributes(row) for row in reader2]
            showTrainResult(tree, test, 9)

def showTrainResult(tree, tests, dimension):
    for k in [1, 5, 10, 100]:
        accuracy = 0
        for query in tests:
            expected = query[-1]
            query = query[1:-1]
            ans = knnClassifier(tree, k, query, dimension)
            if expected in ans:
                accuracy += 1.0 / len(ans)
            # print (ans)
        accuracy /= len(tests)
        print ("KNN accuracy: %f" % accuracy)
        for i in range(3):
            query = tests[i][1:-1]
            result = knn(tree, k, query, dimension)
            # result is a list of tuples (distance, record)
            print (" ".join([str(rec[1][0]) for rec in result]))
        print ("")
main()
