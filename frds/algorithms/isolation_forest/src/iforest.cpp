#include "iforest.hpp"

#include <numeric>

IsolationForest::IsolationForest(Data &data, int treeSize, int forestSize,
                                 int randomSeed)
    : maxTreeHeight(ceil(log2(treeSize))), data(data), treeSize(treeSize),
      forestSize(forestSize), randomSeed(randomSeed), nObs(data[0].size())
{
    this->randomGen = std::mt19937_64(randomSeed);
    this->dist = std::uniform_int_distribution<int>(0, this->data.size() - 1);
}

double IsolationForest::anomalyScore(int observationId)
{
    double average = 0.0;
    for (auto &tree : this->trees)
    {
        average +=
            this->pathLength(std::make_shared<IsolationTree>(tree), observationId);
    }
    average /= this->forestSize;
    return pow(2, -average / this->averagePathLengthInBST(this->treeSize));
}

inline double IsolationForest::averagePathLengthInBST(double n)
{
    return 2 * (log(n - 1) + EULER_GAMMA) - (2 * (n - 1) / n);
}

double IsolationForest::pathLength(std::shared_ptr<IsolationTree> tree,
                                   int observationId, int length)
{
    if (tree->isExNode)
    {
        if (tree->nObs <= 1)
            return length;
        return length + this->averagePathLengthInBST(tree->nObs);
    }
    if (this->data[tree->splitAttr][observationId] <= tree->splitVal)
        return this->pathLength(tree->ltree, observationId, length + 1);
    return this->pathLength(tree->rtree, observationId, length + 1);
}

void IsolationForest::growForest()
{
    if (this->data.empty())
        return;

    // Make a vector from 0 to (nObs-1) representing the indices of observations
    std::vector<int> obs(this->nObs);
    std::iota(obs.begin(), obs.end(), 0);
    // Growing the forest
    for (int i = 0; i < this->forestSize; i++)
    {
        // Sample `treeSize` observations without replacement
        std::vector<int> sample;
        std::sample(obs.begin(), obs.end(), std::back_inserter(sample),
                    this->treeSize, this->randomGen);
        // Add the tree to the forest
        this->trees.push_back(this->growTree(sample));
    }
}

IsolationTree IsolationForest::growTree(std::vector<int> obs,
                                        int currentHeight)
{
    IsolationTree tree;
    tree.nObs = obs.size();

    if ((tree.nObs <= 1) || (currentHeight >= this->maxTreeHeight))
    {
        tree.isExNode = true;
        return tree;
    }
    tree.splitAttr = this->dist(this->randomGen);

    // TODO: uniform distribution [min, max] of values?
    std::uniform_int_distribution<int> dist(0, obs.size() - 1);
    tree.splitVal = this->data[tree.splitAttr][obs[dist(this->randomGen)]];

    std::vector<int> lobs, robs;
    for (auto &i : obs)
    {
        if (this->data[tree.splitAttr][i] <= tree.splitVal)
        {
            lobs.push_back(i);
        }
        else
        {
            robs.push_back(i);
        }
    }
    tree.ltree =
        std::make_shared<IsolationTree>(this->growTree(lobs, currentHeight + 1));
    tree.rtree =
        std::make_shared<IsolationTree>(this->growTree(robs, currentHeight + 1));
    return tree;
}

std::vector<double> calAnomalyScore(Data data, int forestSize, int treeSize,
                                    int randomSeed)
{
    auto iforest = IsolationForest(data, treeSize, forestSize, randomSeed);
    iforest.growForest();
    std::vector<double> ascores;
    for (int i = 0; i < iforest.nObs; i++)
        ascores.push_back(iforest.anomalyScore(i));
    return ascores;
}
