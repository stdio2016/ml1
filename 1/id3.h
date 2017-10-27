// can only handle continuous features
struct decision_tree {
  int target; // -1 means not leaf node, >= 0 means leaf
  int feature;
  float threashold;
  struct decision_tree *less, *more;
};
extern int Target[];
extern float Features[][150];

void shuffleData();
