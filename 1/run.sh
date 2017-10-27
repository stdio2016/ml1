#wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data
gcc -o main main.c id3.c -lm -std=c99
./main
