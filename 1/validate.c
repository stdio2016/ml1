#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include "id3.h"
#include "validate.h"

struct id3_performance_t {
  float accuracy;
  float precision[3];
  float recall[3];
} Perf[5];

int predictFromTree(int dataId, struct decision_tree *tree) {
  if (tree->target >= 0) return tree->target;
  if (Features[tree->feature][dataId] > tree->threashold) {
    return predictFromTree(dataId, tree->more);
  }
  return predictFromTree(dataId, tree->less);
}

// k-fold validation where k=5
void *kfoldRunner(void *param){
  int fold = *(int *) param;
  const int k = 5, foldsize = 150 / k;
  int *training = malloc(sizeof(int) * (150 - foldsize));
  int *validate = malloc(sizeof(int) * foldsize);
  if (training == NULL) exit(2);
  int i, j, n = 0;
  for (i = 0; i < k; i++) {
    if (i != fold)
    for (j = 0; j < foldsize; j++) {
      training[n++] = i * foldsize + j;
    }
  }
  for (j = 0; j < foldsize; j++) {
    validate[j] = fold * foldsize + j;
  }
  int allfeatures[4] = {0, 1, 2, 3};
  struct decision_tree *tree = id3_from_data(training, 150 - foldsize, 4, allfeatures);
  int correct[3] = {0,0,0};
  int count[3] = {0,0,0};
  int predcount[3] = {0,0,0};
  for (i = 0; i < foldsize; i++) {
    int predict = predictFromTree(validate[i], tree);
    int actual = Target[validate[i]];
    if (predict == actual) correct[actual]++;
    count[actual]++;
    predcount[predict]++;
  }
  int correctSum = 0;
  for (j = 0; j < 3; j++) {
    correctSum += correct[j];
    Perf[fold].precision[j] = correct[j] * 1.0 / predcount[j];
    Perf[fold].recall[j] = correct[j] * 1.0 / count[j];
  }
  Perf[fold].accuracy = correctSum * 1.0 / foldsize;
  free(training);
  free(validate);
  return NULL;
}

void kfold() {
  int params[5] = {0,1,2,3,4};
  pthread_attr_t att;
  pthread_t pid[5];
  pthread_attr_init(&att);
  int i;
  for (i = 0; i < 5; i++) {
    pthread_create(&pid[i], &att, kfoldRunner, &params[i]);
  }
  pthread_attr_destroy(&att);
  struct id3_performance_t sum = {};
  for (i = 0; i < 5; i++) {
    pthread_join(pid[i], NULL);
    sum.accuracy += Perf[i].accuracy;
    int j;
    for (j = 0; j < 3; j++) {
      sum.precision[j] += Perf[i].precision[j];
      sum.recall[j] += Perf[i].recall[j];
    }
  }
  printf("%f\n", sum.accuracy / 5);
  for (i = 0; i < 3; i++) {
    printf("%f %f\n", sum.precision[i] / 5, sum.recall[i] / 5);
  }
}
