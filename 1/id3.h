#define DATASIZE 150
#define CLASSCOUNT 3
#define FEATURECOUNT 4
// can only handle continuous features
struct decision_tree {
  int target; // -1 means not leaf node, >= 0 means leaf
  int feature;
  float threashold;
  struct decision_tree *less, *more;
};
extern int Target[];
extern float Features[][DATASIZE];

void shuffleData();
struct decision_tree *id3_from_data(const int *dataIds, int size, int featureCount, int *features);
void printDecision(struct decision_tree *node, int indent);
void freeTree(struct decision_tree *tree);
