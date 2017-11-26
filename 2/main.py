import csv
import sys
import collections
import math
import heapq

KdtreeNode = collections.namedtuple('KdtreeNode', ['center', 'left', 'right', 'size'])

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
        # showKdtree(Kdtree(dat, 9, 0))
        show = (knn(Kdtree(dat, 9, 0), 10, [0,0,0,0,0,0,0,0,0], 9))
        for i in show:
             print (i[0])

def Kdtree(points, k, depth):
    n = len(points)
    if n == 0:
        return None

    # select axis by depth
    axis = depth % k
    # sort by axis
    A = sorted(points, key=lambda x: x[axis+1])

    middle = n//2
    return KdtreeNode(
        size = n,
        center = tuple(A[middle]),
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
def knn(tree, k, query, dimension):
    bests = [] # bests is a heap which stores tuple (-distance, record)
    knn_helper(tree, k, query, dimension, 0, bests, ())
    result = []
    for i in range(k):
        result.append((-bests[0][0], bests[0][1]))
        heapq.heappop(bests)
    result.reverse()
    return result

# helper for knn
# node: a node of kdtree
# dim: dimension
# lv: level of recursion
# bests: current best
def knn_helper(node, k, query, dim, lv, bests, bounds):
    if node == None:
        return None
    axis = lv % dim
    if query[axis] > node.center[axis+1]:
        good = node.right
        bad = node.left
    else:
        good = node.left
        bad = node.right
    knn_helper(good, k, query, dim, lv+1, bests, bounds)
    d = mydistance(query, node.center[1:])
    if len(bests) < k:
        heapq.heappush(bests, (-d, node.center))
    else:
        if -bests[0][0] > d: # distance too large
            heapq.heappop(bests)
            heapq.heappush(bests, (-d, node.center))
        pass
    # check if another side is possible
    bounds = bounds[:axis] + (node.center[axis+1] - query[axis],) + bounds[axis+1:]
    mindist = 0
    for i in bounds:
        mindist += i * i
    mindist = math.sqrt(mindist)
    if mindist < -bests[0][0]: # another side is possible
        knn_helper(bad, k, query, dim, lv+1, bests, bounds)
    return None

def mydistance(vec1, vec2):
    s = 0
    for i in range(len(vec1)):
        s += (vec2[i] - vec1[i]) ** 2
    return math.sqrt(s)

main()
