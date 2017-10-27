#include <math.h>
#include <stdlib.h>
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
  subcount[prevTarget]++;
  int targetChangePrev = 0;
  float best = -1000; // should be -Infinite
  for (i = from + 1; i < to; i++) {
    int id = sortOrder[i];
    int t = Target[id];
    float f = feature[id];
    float prevf = feature[id - 1];
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
  }
  else {
    node->target = -1;
    node->feature = feature;
    int *sortOrder = &state->sortOrder[150 * feature];
    int pless = sortOrder[partition - 1], pmore = sortOrder[partition];
    float threashold = (Features[feature][pless] + Features[feature][pmore]) / 2;
    node->threashold = threashold;
    // arrange training data
    int *temp = state->temp; // get temp storage
    int less = from, more = partition;
    for (i = 0; i < 4; i++) { // for each feature
      int j;
      if (i == feature) continue; //no need to sort that!
      int *sortOrder = &state->sortOrder[150 * i];
      for (j = from; j < to; j++) {
        if (Features[i][sortOrder[j]] < threashold) {
          temp[less++] = sortOrder[j];
        }
        else {
          temp[more++] = sortOrder[j];
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
    qsort(&sortOrder[i * 150], sizeof(int), 150, featureCompare);
  }
  return sortOrder;
}
