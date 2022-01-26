#ifndef FRDS_ALGO_ISOLATION_FOREST_H
#define FRDS_ALGO_ISOLATION_FOREST_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <memory>
#include <random>
#include <vector>

#include "numpy/arrayobject.h"

typedef double DataType;
typedef char *CharDataType;

class IsolationTree {
 private:
  struct Node {
    std::unique_ptr<Node> lnode, rnode;
    const size_t splitAttribute;
    const DataType splitValue;
    const CharDataType splitChar;
    const bool isExNode;
    const int nObs;

    Node(size_t splitAttribute, DataType splitValue, CharDataType splitChar,
         bool isExNode, int nObs)
        : splitAttribute(splitAttribute),
          splitValue(splitValue),
          splitChar(splitChar != nullptr ? splitChar : ""),
          isExNode(isExNode),
          nObs(nObs),
          lnode(nullptr),
          rnode(nullptr) {}
  };

  typedef std::unique_ptr<Node> ptrNode;
  ptrNode root;

 public:
  IsolationTree() = default;
  ~IsolationTree() = default;

  friend class IsolationForest;
};

class IsolationForest {
 private:
  /* data */
  PyArrayObject *num_data, *char_data;
  const size_t treeSize, forestSize, randomSeed, maxTreeHeight;
  const size_t n_num_attrs, n_char_attrs, nObs;
  std::vector<std::unique_ptr<IsolationTree>> trees;
  std::mt19937_64 randomGen;
  std::uniform_int_distribution<size_t> uniformDist;

  /* methods */
  inline double averagePathLength(size_t const &nObs);
  double pathLength(size_t const &ob,
                    std::unique_ptr<IsolationTree::Node> &node, int length = 0);
  void growTree(std::vector<size_t> &sample,
                std::unique_ptr<IsolationTree::Node> &node,
                int const height = 0);

 public:
  IsolationForest(PyArrayObject *num_data, PyArrayObject *char_data,
                  size_t const &treeSize = 256,
                  size_t const &forestSize = 1'000,
                  size_t const &randomSeed = 1);
  ~IsolationForest() = default;
  void growForest();
  double anomalyScore(size_t const &ob);
};

#endif  // FRDS_ALGO_ISOLATION_FOREST_H