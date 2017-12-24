import csv
import math
import numpy as np

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

