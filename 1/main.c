#include <stdio.h>
#include <string.h>
#include "id3.h"
#include "validate.h"

void mlread(FILE *f)
{
  int i;
  for (i = 0; i < 150; i++) {
    float a, b, c, d;
    fscanf(f, "%f,%f,%f,%f,", &a, &b, &c, &d);
    Features[0][i] = a;
    Features[1][i] = b;
    Features[2][i] = c;
    Features[3][i] = d;
    char cls[100];
    fgets(cls, 100 ,f);
    int t = 0;
    if (strcmp(cls, "Iris-setosa\n") == 0) {
      t = 0;
    }
    else if (strcmp(cls, "Iris-versicolor\n") == 0) {
      t = 1;
    }
    else if (strcmp(cls, "Iris-virginica\n") == 0) {
      t = 2;
    }
    Target[i] = t;
  }
}

int main(int argc, char *argv[])
{
  FILE *f;
  f = fopen("bezdekIris.data", "r");
  if (f == NULL) {
    printf("cannot read data\n");
  }
  mlread(f);
  shuffleData();
  if (argc == 2 && strcmp(argv[1], "-RF") == 0) {\
    randomForest();
  }
  else {
    kfold();
  }
  fclose(f);
  return 0;
}
