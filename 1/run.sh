#wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data
gcc -o main main.c id3.c validate.c -lm -std=c99 -lpthread
./main
