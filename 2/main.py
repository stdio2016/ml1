import csv
import sys

def main():
    if (len(sys.argv) < 3):
        print ("Usage: ./run.sh [train data] [test data]")
        sys.exit()
    traincsv = sys.argv[1]
    testcsv = sys.argv[2]
    with open(traincsv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip first line
        dat = [row for row in reader]
        Kdtree(dat, 9, 0)

class Kdtree:
    def __init__(self, points, k, depth):
        n = len(points)
        self.size = n
        if n == 1:
            self.center = points[0]
            self.left = None
            self.right = None
            return

        # select axis by depth
        axis = depth % k
        # sort by axis
        A = sorted(points, key=lambda x: x[axis])

        middle = n//2
        self.center = A[middle]
        self.left = Kdtree(A[0:middle], k, depth+1)
        # boundary case
        if n <= 2:
            self.right = None
        else:
            self.right = Kdtree(A[middle+1:], k, depth+1)

main()
