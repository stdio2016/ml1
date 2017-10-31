#include <stdlib.h>
#include <stdio.h>
#ifdef PARALLEL
#include <pthread.h>
#endif
#include "id3.h"
#include "validate.h"

struct id3_performance_t {
  float accuracy;
  float precision[CLASSCOUNT];
  float recall[CLASSCOUNT];
} Perf[KFOLD_K];

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
  int foldsize = DATASIZE / KFOLD_K;
  int *training = malloc(sizeof(int) * (DATASIZE - foldsize));
  int *validate = malloc(sizeof(int) * foldsize);
  if (training == NULL) exit(2);
  int i, j, n = 0;
  for (i = 0; i < KFOLD_K; i++) {
    if (i != fold)
    for (j = 0; j < foldsize; j++) {
      training[n++] = i * foldsize + j;
    }
  }
  for (j = 0; j < foldsize; j++) {
    validate[j] = fold * foldsize + j;
  }
  int allfeatures[FEATURECOUNT];
  for (i = 0; i < FEATURECOUNT; i++) {
    allfeatures[i] = i;
  }
  struct decision_tree *tree = id3_from_data(training, DATASIZE - foldsize, FEATURECOUNT, allfeatures);
  int correct[CLASSCOUNT] = {0,0,0};
  int count[CLASSCOUNT] = {0,0,0};
  int predcount[CLASSCOUNT] = {0,0,0};
  for (i = 0; i < foldsize; i++) {
    int predict = predictFromTree(validate[i], tree);
    int actual = Target[validate[i]];
    if (predict == actual) correct[actual]++;
    count[actual]++;
    predcount[predict]++;
  }
  int correctSum = 0;
  for (j = 0; j < CLASSCOUNT; j++) {
    correctSum += correct[j];
    Perf[fold].precision[j] = correct[j] * 1.0 / predcount[j];
    Perf[fold].recall[j] = correct[j] * 1.0 / count[j];
  }
  Perf[fold].accuracy = correctSum * 1.0 / foldsize;
  free(training);
  free(validate);
  freeTree(tree);
  return NULL;
}

void kfold() {
  int params[KFOLD_K];
  int i;
  for (i = 0; i < KFOLD_K; i++) {
    params[i] = i;
  }
#ifdef PARALLEL
  pthread_attr_t att;
  pthread_t pid[KFOLD_K];
  pthread_attr_init(&att);
  for (i = 0; i < KFOLD_K; i++) {
    pthread_create(&pid[i], &att, kfoldRunner, &params[i]);
  }
  pthread_attr_destroy(&att);
#endif
  struct id3_performance_t sum = {};
  for (i = 0; i < KFOLD_K; i++) {
#ifdef PARALLEL
    pthread_join(pid[i], NULL);
#else
    kfoldRunner(&params[i]);
#endif
    sum.accuracy += Perf[i].accuracy;
    int j;
    for (j = 0; j < CLASSCOUNT; j++) {
      sum.precision[j] += Perf[i].precision[j];
      sum.recall[j] += Perf[i].recall[j];
    }
  }
  printf("%.3f\n", sum.accuracy / KFOLD_K);
  for (i = 0; i < CLASSCOUNT; i++) {
    printf("%.3f %.3f\n", sum.precision[i] / KFOLD_K, sum.recall[i] / KFOLD_K);
  }
}
