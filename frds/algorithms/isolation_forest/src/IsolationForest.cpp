#include "IsolationForest.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>

#ifndef EULER_GAMMA
#define EULER_GAMMA 0.57721566490153286
#endif

void IsolationForest::growTree(std::vector<size_t> &sample,
                               std::unique_ptr<IsolationTree::Node> &node,
                               int const height) {
  auto nObs = sample.size();
  if ((nObs <= 1) || (height >= this->maxTreeHeight)) {
    node = std::make_unique<IsolationTree::Node>(-1, NAN, true, nObs);
    return;
  }
  auto attr = this->uniformDist(this->randomGen);

  std::uniform_int_distribution<size_t> dist(0, nObs - 1);
  DataType val = *(DataType *)PyArray_GETPTR2(this->data, attr,
                                              sample[dist(this->randomGen)]);

  std::vector<size_t> lobs, robs;
  for (auto &i : sample) {
    auto obsVal = *(DataType *)PyArray_GETPTR2(this->data, attr, i);
    // NAN is set to be "smaller" than any value
    // If the split value is NAN, then obs with NAN are pushed left
    if (isnan(val)) {
      if (isnan(obsVal)) {
        lobs.push_back(i);
      } else {
        robs.push_back(i);
      }
      // Split value is not NAN, then obs with NAN or smaller are pushed left
    } else {
      if ((isnan(obsVal)) || (obsVal <= val)) {
        lobs.push_back(i);
      } else {
        robs.push_back(i);
      }
    }
  }
  node = std::make_unique<IsolationTree::Node>(attr, val, false, nObs);
  growTree(lobs, node->lnode, height + 1);
  growTree(robs, node->rnode, height + 1);
}

IsolationForest::IsolationForest(PyArrayObject *data, size_t const &treeSize,
                                 size_t const &forestSize,
                                 size_t const &randomSeed)
    : treeSize(treeSize),
      forestSize(forestSize),
      randomSeed(randomSeed),
      maxTreeHeight(ceil(log2((double)treeSize))),
      nAttrs(PyArray_DIM(data, 0)),
      nObs(PyArray_DIM(data, 1)) {
  this->data = data;
  this->trees.reserve(forestSize);
  this->randomGen = std::mt19937_64(randomSeed);
  this->uniformDist = std::uniform_int_distribution<size_t>(0, nAttrs - 1);
}

double IsolationForest::averagePathLength(size_t const &nObs) {
  auto n = (double)nObs;
  return 2 * (log(n - 1) + EULER_GAMMA) - (2 * (n - 1) / n);
}

void IsolationForest::growForest() {
  // Make a vector from 0 to (nObs-1) representing the indices of observations
  std::vector<size_t> obs(this->nObs);
  std::iota(obs.begin(), obs.end(), 0);

  for (size_t i = 0; i < this->forestSize; i++) {
    // Sample `treeSize` observations without replacement
    std::vector<size_t> sample;
    std::sample(obs.begin(), obs.end(), std::back_inserter(sample),
                this->treeSize, this->randomGen);

    auto tree = std::make_unique<IsolationTree>();
    this->growTree(sample, tree->root);
    this->trees.push_back(std::move(tree));
  }
}

double IsolationForest::anomalyScore(size_t const &ob) {
  double avg = 0;
  for (auto &tree : this->trees) avg += this->pathLength(ob, tree->root);
  avg /= this->forestSize;
  return pow(2, -avg / this->averagePathLength(this->treeSize));
}

double IsolationForest::pathLength(size_t const &ob,
                                   std::unique_ptr<IsolationTree::Node> &node,
                                   int length) {
  if (node->isExNode) {
    if (node->nObs <= 1) return length;
    return length + this->averagePathLength(node->nObs);
  }
  DataType val =
      *(DataType *)PyArray_GETPTR2(this->data, node->splitAttribute, ob);
  if (val < node->splitValue) {
    return this->pathLength(ob, node->lnode, length + 1);
  } else {
    return this->pathLength(ob, node->rnode, length + 1);
  }
}