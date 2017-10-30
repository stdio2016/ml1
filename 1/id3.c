#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "id3.h"
int Target[150];
float Features[4][150];
// can only handle continuous features

struct id3_state {
  int featureCount;
  int *features; // features that this tree sees
  int *sortOrder;
  int *temp; // temporary storage of size 150
};

float logcache[151];

// Calculate information gain
// sortOrder[from ~ to-1] is dataset D, sorted by descriptive feature feat
// H is entropy of the dataset D
float ig(const int *sortOrder, int feat, int from, int to, const int *count, int *partition, float H)
{
  int i;
  int subcount[3] = {0,0,0};
  int prevTarget = Target[sortOrder[from]];
  float *feature = Features[feat];
  float prevf = feature[sortOrder[from]];
  subcount[prevTarget]++;
  int targetChangePrev = 0;
  for (i = from + 1; i < to; i++) {
    // maybe target changes but feature doesn't
    int id = sortOrder[i];
    if (feature[id] != prevf) break;
    if (Target[id] != prevTarget) {
      targetChangePrev = 1;
      break;
    }
  }
  float best = -INFINITY;
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
        // compute information gain = H(t,D) - rem(d,D)
        int j;
        float gain = H;
        // compute rem(d,D)
        float Hless = 0, Hmore = 0;
        for (j = 0; j < 3; j++) {
          if (subcount[j] == 0) continue;
          float P = 1.0*subcount[j] / (i - from);
          Hless -= P * log2(P);
        }
        for (j = 0; j < 3; j++) {
          if (count[j] == subcount[j]) continue;
          float P = 1.0*(count[j] - subcount[j]) / (to - i);
          Hmore -= P * log2(P);
        }
        gain -= (Hless * (i - from) + Hmore * (to - i)) / (to - from);
        if (gain > best) {
          best = gain;
          *partition = i;
        }
      }
      targetChangePrev = targetChange;
    }
    subcount[t]++;
    prevTarget = t;
    prevf = f;
  }
  return best;
}

// build decision tree
// from~to-1: range of data
struct decision_tree *id3_runner(struct id3_state *state, int from, int to)
{
  struct decision_tree *node = malloc(sizeof(struct decision_tree));
  if (node == NULL) {
    exit(-2);
  }
  // count each target
  int count[3] = {0,0,0};
  int i;
  for (i = from; i < to; i++) {
    count[Target[state->sortOrder[i]]]++;
  }
  // compute entropy of whole dataset
  float H = 0;
  for (i = 0; i < 3; i++) {
    if (count[i] == 0) continue;
    float P = 1.0*count[i] / (to - from);
    H -= P * log2(P);
  }
  // find feature with most entropy
  int partition = -1, feature = 0;
  float entropy = ig(&state->sortOrder[0], state->features[0], from, to, count, &partition, H);
  for (i = 1; i < state->featureCount; i++) {
    int pa = -1;
    float en = ig(&state->sortOrder[150 * i], state->features[i], from, to, count, &pa, H);
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
    node->feature = state->features[feature];
    int *sortOrder = &state->sortOrder[150 * feature];

    feature = state->features[feature];
    // calculate threshold
    int pless = sortOrder[partition - 1], pmore = sortOrder[partition];
    float threashold = (Features[feature][pless] + Features[feature][pmore]) / 2;
    node->threashold = threashold;
    // arrange training data
    int *temp = state->temp; // get temp storage
    for (i = 0; i < state->featureCount; i++) { // for each feature
      int j;
      if (i == feature) continue; //no need to sort that!
      int *sortOrder = &state->sortOrder[150 * i];
      int less = from, more = partition;
      for (j = from; j < to; j++) {
        if (Features[feature][sortOrder[j]] < threashold) {
          temp[less++] = sortOrder[j];
        }
        else {
          temp[more++] = sortOrder[j];
        }
      }
      for (j = from; j < to; j++) {
        sortOrder[j] = temp[j];
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
  srand(time(NULL));
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

void sortByFeature(const int *dataIds, int size, int feature, int *out, int *tmp) {
  float *feats = Features[feature];
  if (size < 10) {
    // bubble sort
    int i, j;
    for (i = 0; i < size; i++) {
      out[i] = dataIds[i];
    }
    for (i = 0; i < size; i++) {
      for (j = 1; j < size; j++) {
        if (feats[out[j-1]] > feats[out[j]]) {
          int t = out[j];
          out[j] = out[j-1];
          out[j-1] = t;
        }
      }
    }
  }
  else {
    // merge sort
    int half = size>>1;
    sortByFeature(dataIds, half, feature, out, tmp);
    sortByFeature(dataIds + half, size - half, feature, out + half, tmp);
    int a = 0, b = half, c = 0;
    while (a < half && b < size) {
      if (feats[out[a]] > feats[out[b]]) {
        tmp[c] = out[b++];
      }
      else {
        tmp[c] = out[a++];
      }
      c++;
    }
    while (a < half) tmp[c++] = out[a++];
    while (b < size) tmp[c++] = out[b++];
    for (c = 0; c < size; c++) {
      out[c] = tmp[c];
    }
  }
}

// build decision tree from some data
struct decision_tree *id3_from_data(const int *dataIds, int size, int featureCount, int *features) {
  struct id3_state state;
  state.featureCount = featureCount;
  state.features = features;
  state.sortOrder = malloc(sizeof(int) * 150 * featureCount);
  state.temp = malloc(sizeof(int) * 150);
  int i;
  for(i = 0; i < featureCount; i++) {
    sortByFeature(dataIds, size, features[i], &state.sortOrder[150 * i], state.temp);
  }
  struct decision_tree *tree = id3_runner(&state, 0, size);
  free(state.sortOrder);
  free(state.temp);
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

void freeTree(struct decision_tree *tree) {
  if (tree->target < 0) {
    freeTree(tree->less);
    freeTree(tree->more);
    free(tree);
  }
  else { // leaf node
    free(tree);
  }
}
