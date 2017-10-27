#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "id3.h"
int Target[150];
float Features[4][150];
// can only handle continuous features

struct id3_state {
  int *sortOrder;
  int *temp; // temporary storage of size 150
};

float logcache[151];

float ig(const int *sortOrder, int feat, int from, int to, const int *count, int *partition)
{
  int i;
  int subcount[3] = {0,0,0};
  int prevTarget = Target[sortOrder[from]];
  float *feature = Features[feat];
  float prevf = feature[sortOrder[from]];
  subcount[prevTarget]++;
  int targetChangePrev = 0;
  float best = -1000; // should be -Infinite
  for (i = from + 1; i < to; i++) {
    int id = sortOrder[i];
    int t = Target[id];
    float f = feature[id];
    if (f != prevf) { // possible partition point
      // maybe target changes but feature doesn't
      int targetChange = 0;
      int j;
      for (j = i + 1; j < to; j++) {
        id = sortOrder[j];
        if (feature[id] != f) break;
        if (Target[id] != t) {
          targetChange = 1;
          break;
        }
      }
      // a partition point
      if (targetChange || targetChangePrev || t != prevTarget) {
        // compute entropy
        float entropy = 0;
        int j;
        // for each target
        for (j = 0; j < 3; j++) {
          entropy += logcache[subcount[j]] + logcache[count[j] - subcount[j]];
        }
        entropy -= logcache[i - from] + logcache[to - i];
        if (entropy > best) {
          best = entropy;
          *partition = i;
        }
      }
      targetChangePrev = targetChange;
    }
    subcount[t]++;
    prevTarget = t;
    prevf = f;
  }
  return best / (to - from);
}

struct decision_tree *id3_runner(struct id3_state *state, int from, int to)
{
  struct decision_tree *node = malloc(sizeof(struct decision_tree));
  if (node == NULL) {
    exit(-2);
  }
  int count[3] = {0,0,0};
  int i;
  for (i = from; i < to; i++) {
    count[Target[state->sortOrder[i]]]++;
  }
  // find feature with most entropy
  int partition = -1, feature = 0;
  float entropy = ig(&state->sortOrder[0], 0, from, to, count, &partition);
  for (i = 1; i < 4; i++) {
    int pa = -1;
    float en = ig(&state->sortOrder[150 * i], i, from, to, count, &pa);
    if (en > entropy) {
      partition = pa;
      entropy = en;
      feature = i;
    }
  }
  if (partition == -1) {
    // cannot make partition => create leaf node
    int majority = 0;
    for (i = 1; i < 3; i++) {
      if (count[i] > count[majority]) {
        majority = i;
      }
    }
    node->target = majority;
    node->feature = to -from;
  }
  else {
    node->target = -1;
    node->feature = feature;
    int *sortOrder = &state->sortOrder[150 * feature];
    int pless = sortOrder[partition - 1], pmore = sortOrder[partition];
    float threashold = (Features[feature][pless] + Features[feature][pmore]) / 2;
    node->threashold = threashold;
    // arrange training data
    for (i = 0; i < 4; i++) { // for each feature
      if (i == feature) continue; //no need to sort that!
      int *sortOrder = &state->sortOrder[150 * i];
      int less = from, more = to - 1;
      while (less < more) {
        while (less < to - 1 && Features[i][sortOrder[less]] < threashold) {
          less++;
        }
        while (more > from && Features[i][sortOrder[more]] >= threashold) {
          more--;
        }
        if (less < more) {
          int tmp = sortOrder[less];
          sortOrder[less] = sortOrder[more];
          sortOrder[more] = tmp;
          less++; more--;
        }
      }
    }
    // now call id3 recursively
    node->less = id3_runner(state, from, partition);
    node->more = id3_runner(state, partition, to);
  }
  return node;
}

void swapfloat(float *a, float *b) {
  float tmp = *a;
  *a = *b;
  *b = tmp;
}

void shuffleData() {
  srand(9487);
  int i, r;
  for (i = 0; i < 150-1; i++) {
    r = rand() % (150 - i) + i;
    if (r == i) continue;
    // swap item at r and i
    int tmp = Target[i];
    Target[i] = Target[r];
    Target[r] = tmp;
    int j = 0;
    for (j = 0; j < 4; j++) {
      swapfloat(&Features[j][i], &Features[j][r]);
    }
  }
}

static int featureToComp;
int featureCompare(const void *a, const void *b) {
  int i = *(int *) a, j = *(int *) b;
  float fa = Features[featureToComp][i];
  float fb = Features[featureToComp][j];
  if (fa < fb) return -1;
  if (fa == fb) return 0;
  if (fa > fb) return 1;
}

int *sortFeatures() {
  int i, j;
  int *sortOrder = malloc(sizeof(int) * 150 * 4);
  if (sortOrder == NULL) exit(-2);
  for (i = 0; i < 4; i++) {
    for (j = 0; j < 150; j++) {
      sortOrder[i * 150 + j] = j;
    }
  }
  for (i = 0; i < 4; i++) {
    featureToComp = i;
    qsort(&sortOrder[i * 150], 150, sizeof(int), featureCompare);
  }
  return sortOrder;
}

struct decision_tree *id3_for_all() {
  int i;
  logcache[0] = 0;
  for (i = 1; i <= 150; i++) {
    logcache[i] = i * log(i);
  }
  int *order = sortFeatures();
  struct id3_state state;
  state.sortOrder = order;
  struct decision_tree *tree = id3_runner(&state, 0, 150);
  free(order);
  return tree;
}

void printDecision(struct decision_tree *node, int indent) {
  int i;
  for (i = 0; i < indent; i++) {
    printf(" ");
  }
  if (node->target == 0) {
    printf("Iris-setosa %d\n", node->feature);
  }
  else if (node->target == 1) {
    printf("Iris-versicolor %d\n", node->feature);
  }
  else if (node->target == 2) {
    printf("Iris-virginica %d\n", node->feature);
  }
  else {
    printf("test feature %d, threashold = %f\n", node->feature, node->threashold);
    printDecision(node->less, indent+1);
    printDecision(node->more, indent+1);
  }
}

