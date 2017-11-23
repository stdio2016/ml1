import csv
import sys
import collections

KdtreeNode = collections.namedtuple('KdtreeNode', ['center', 'left', 'right', 'size'])

# only some of the columns in csv are used
# and I need to convert string to number
def selectAttributes(row):
    selected = []
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
        showKdtree(Kdtree(dat, 9, 0))

def Kdtree(points, k, depth):
    n = len(points)
    if n == 0:
        return None

    # select axis by depth
    axis = depth % k
    # sort by axis
    A = sorted(points, key=lambda x: x[axis])

    middle = n//2
    return KdtreeNode(
        size = n,
        center = A[middle],
        left = Kdtree(A[:middle], k, depth+1),
        right = Kdtree(A[middle+1:], k, depth+1)
    )

def showKdtree(tree, lv=0):
    if tree == None:
        print ("    " * lv + "None")
    else:
        print ("    " * lv + str(tree.center))
        showKdtree(tree.left, lv+1)
        showKdtree(tree.right, lv+1)

# find k nearest neighbors of the query point
def knn(k, query, dimension):
    return 

main()
