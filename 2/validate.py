import csv
import sys
from kdtree import knn, Kdtree

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

def knnValidate(tree, k, query, dimension):
    neighbor = knn(tree, k+1, query, dimension)
    s = {}
    for poll in neighbor:
        if poll[0] == 0:
            continue
        cls = poll[3]
        if cls in s:
            s[cls] += 1
        else:
            s[cls] = 1
    mx = 0
    ans = []
    for u in s:
        if s[u] > mx:
            mx = s[u]
            ans = [u]
        elif s[u] == mx:
            ans.append(u)
    return ans

def showValidateResult(data, dimension):
    tree = Kdtree(data, 9, 0)
    for k in range(1, 101):
        print ("k = %d" % k)
        accuracy = 0
        for query in data:
            expected = query[-1]
            query = query[1:-1]
            ans = knnValidate(tree, k, query, dimension)
            if expected in ans:
                accuracy += 1.0 / len(ans)
            # print (ans)
        accuracy /= len(data)
        print ("KNN accuracy: %f" % accuracy)
main()
