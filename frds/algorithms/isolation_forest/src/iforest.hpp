#ifndef Py_IFOREST_MODULE_H
#define Py_IFOREST_MODULE_H

#include <math.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#ifndef EULER_GAMMA
#define EULER_GAMMA 0.57721566490153286
#endif

typedef double DataType;
typedef std::vector<std::vector<DataType>> Data;

class IsolationTree
{
private:
    std::shared_ptr<IsolationTree> ltree, rtree;
    int nObs;
    int splitAttr;
    DataType splitVal;
    bool isExNode = false;

    friend class IsolationForest;

public:
    IsolationTree(){};
    ~IsolationTree(){};
};

class IsolationForest
{
private:
    const Data data;
    const int forestSize;
    const int treeSize;
    const int randomSeed;
    const int maxTreeHeight;
    const int nObs;
    std::mt19937_64 randomGen;
    std::uniform_int_distribution<int> dist;
    std::vector<IsolationTree> trees = {};
    double averagePathLengthInBST(double n);
    double pathLength(std::shared_ptr<IsolationTree> tree, int observationId,
                      int length = 0);
    IsolationTree growTree(std::vector<int> obs, int currentHeight = 0);

public:
    IsolationForest(Data &data, int treeSize = 256, int forestSize = 1000,
                    int randomSeed = 1);
    ~IsolationForest(){};
    void growForest();
    double anomalyScore(int observationId);

    friend std::vector<double> calAnomalyScore(Data data, int forestSize,
                                               int treeSize, int randomSeed);
};

std::vector<double> calAnomalyScore(Data data, int forestSize, int treeSize,
                                    int randomSeed);

#endif /* !defined(Py_IFOREST_MODULE_H) */