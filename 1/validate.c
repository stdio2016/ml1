#include <stdlib.h>
#include <stdio.h>
#ifdef PARALLEL
#include <pthread.h>
#endif
#include "id3.h"
#include "validate.h"

#define RANDOMFOREST_TREE 10

struct id3_performance_t {
  float accuracy;
  float precision[CLASSCOUNT];
  float recall[CLASSCOUNT];
} Perf[KFOLD_K];

int predictFromTree(int dataId, struct decision_tree *tree) {
  if (tree->target >= 0) return tree->target;
  if (Features[tree->feature][dataId] > tree->threshold) {
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

void kfoldBased(void *(*runner)(void *)) {
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
    pthread_create(&pid[i], &att, runner, &params[i]);
  }
  pthread_attr_destroy(&att);
#endif
  struct id3_performance_t sum = {};
  for (i = 0; i < KFOLD_K; i++) {
#ifdef PARALLEL
    pthread_join(pid[i], NULL);
#else
    runner(&params[i]);
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

int predictFromForest(int dataId, struct decision_tree **forest, int treeCount) {
  int i;
  int count[CLASSCOUNT] = {0};
  for (i = 0; i < treeCount; i++) {
    int ans = predictFromTree(dataId, forest[i]);
    count[ans]++;
  }
  // find best result
  int max = 0, maxtime = 1;
  for (i = 1; i < CLASSCOUNT; i++) {
    if (count[i] > count[max]) {
      maxtime = 1;
      max = i;
    }
    else if (count[i] == count[max]) {
      maxtime++;
    }
  }
  if (maxtime == 1) return max;
  // there are more than one good results
  // first tree with good result wins
  for (i = 0; i < treeCount; i++) {
    int ans = predictFromTree(dataId, forest[i]);
    if (count[ans] == count[max]) {
      return ans;
    }
  }
  return max;
}

void *randomForestRunner(void *param) {
  int fold = *(int *)param;
  int foldsize = DATASIZE / KFOLD_K;
  // test dataset is [testStart, testEnd)
  int testStart = fold * foldsize, testEnd = (fold + 1) * foldsize;
  if (fold == KFOLD_K - 1) {
    testEnd = DATASIZE;
  }
  // construct training dataset
  int size = (DATASIZE - (testEnd - testStart));
  int *training = malloc(sizeof(int) * size);
  int i = 0;
  for (i = 0; i < testStart; i++) {
    training[i] = i;
  }
  for (i = testEnd; i < DATASIZE; i++) {
    training[i - testEnd + testStart] = i;
  }
  // construct forest
  struct decision_tree **forest= malloc(sizeof(struct decision_tree *) * RANDOMFOREST_TREE);
  int *bag = malloc(sizeof(int) * size);
  int allfeatures[FEATURECOUNT];
  for (i = 0; i < FEATURECOUNT; i++) {
    allfeatures[i] = i;
  }
  for (i = 0; i < RANDOMFOREST_TREE; i++) {
    // pick training data, with replacement
    int j;
    for (j = 0; j < size; j++) {
      int r = rand() % size;
      bag[j] = training[r];
    }
    // build a tree
    forest[i] = id3_from_data(bag, size, FEATURECOUNT,  allfeatures);
  }
  // validate
  int correct[CLASSCOUNT] = {0,0,0};
  int count[CLASSCOUNT] = {0,0,0};
  int predcount[CLASSCOUNT] = {0,0,0};
  for (i = testStart; i < testEnd; i++) {
    int predict = predictFromForest(i, forest, RANDOMFOREST_TREE);
    int actual = Target[i];
    if (predict == actual) correct[actual]++;
    count[actual]++;
    predcount[predict]++;
  }
  int correctSum = 0;
  for (i = 0; i < CLASSCOUNT; i++) {
    correctSum += correct[i];
    Perf[fold].precision[i] = correct[i] * 1.0 / predcount[i];
    Perf[fold].recall[i] = correct[i] * 1.0 / count[i];
  }
  Perf[fold].accuracy = correctSum * 1.0 / foldsize;
  // cleanup
  for (i = 0; i < RANDOMFOREST_TREE; i++) {
    freeTree(forest[i]);
  }
  free(forest);
  free(bag);
  free(training);
  return NULL;
}

void kfold(void) {
  kfoldBased(kfoldRunner);
}

void randomForest(void) {
  kfoldBased(randomForestRunner);
}
