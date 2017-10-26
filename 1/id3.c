#include <math.h>
#include "id3.h"
int Target[150];
float Features[4][150];
// can only handle continuous features

struct id3_state {
  int *sortOrder;
  int *temp;
};

float logcache[151];

float ig(const int *sortOrder, int feat, int from, int to, const int *count, int *partition)
{
  int i;
  int subcount[3] = {0,0,0};
  int prevTarget = Target[sortOrder[from]];
  float *feature = Features[feat];
  count[prevTarget]++;
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
    prevtarget = t;
  }
  return best / (to - from);
}

struct decision_tree *id3_runner(struct id3_state *state, int from, int to)
{
  struct decision_tree *node = NULL;
  return node;
}

