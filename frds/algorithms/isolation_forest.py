from math import ceil, log, log2, pow
from dataclasses import dataclass
from typing import List
import random
import pandas as pd
import numpy as np


@dataclass
class IsolationTree:

    ltree: "IsolationTree" = None
    rtree: "IsolationTree" = None
    split_attr: int = None
    split_val: float = None
    n_obs: int = 0
    is_ex_node = False


class IsolationForest:
    def __init__(
        self,
        data: pd.DataFrame,
        tree_size: int = 256,
        forest_size: int = 1_000,
        exclude_cols: list = None,
        random_seed: int = 1,
    ):
        """Isolation Forest

        Args:
            data (pd.DataFrame): indexed DataFrame
            tree_size (int, optional): number of observations per Isolation Tree. Defaults to 256.
            forest_size (int, optional): number of trees. Defaults to 1_000.
            exclude_cols (list, optional): columns in the input `data` to ignore. Defaults to None.
            random_seed (int, optional): random seed for reproducibility. Defaults to 1.
        """
        # set random seed for reproducibility
        random.seed(random_seed)

        # convert input dataset from an indexed `pd.DataFrame` to a `np.ndarray`
        exclude_cols = [] if exclude_cols is None else exclude_cols
        data = data.drop(columns=exclude_cols)
        self.data = data.to_numpy().T

        # an ordered list of column names
        self.attributes = data.columns
        # an ordered list of observation ids
        self.obs = data.index.tolist()
        # a list of all Isolation Trees
        self.trees: List[IsolationTree] = []
        # forest size
        self.forest_size = forest_size

        # tree size is set to the number of observations if it's smaller
        self.tree_size = min(tree_size, len(self.obs))

        # maximum tree height
        self.max_tree_height = ceil(log2(tree_size))

    def anomaly_score(self, observation_id: int) -> float:
        """Calculate the anomaly score of an observation

        Args:
            observation_id (int): index of the observation

        Returns:
            float: anomaly score
        """
        # Average path length across all trees in the forest
        path_length = (
            sum(self.path_length(tree, observation_id) for tree in self.trees)
            / self.forest_size
        )
        return pow(2, -path_length / self.average_path_length_in_bst(self.tree_size))

    def path_length(self, tree: IsolationTree, observation_id: int, length=0) -> float:
        """Calculate the path length of an observation from a given Isolation Tree

        Args:
            tree (IsolationTree): an Isolation Tree
            observation_id (int): index of the observation
            length (int, optional): current path length. Defaults to 0.

        Returns:
            float: the path length
        """
        if tree.is_ex_node:
            if tree.n_obs <= 1:
                # a successful search
                return length
            else:
                # when it's already an ExNode but there are more than 1 observations,
                # we add the average path length of an unsuccessful search in a BST of
                # the same size as the number of remaining observations in the ExNode
                return length + self.average_path_length_in_bst(tree.n_obs)
        else:
            # InNode case, search continues
            if self.data[tree.split_attr][observation_id] <= tree.split_val:
                return self.path_length(tree.ltree, observation_id, length + 1)
            else:
                return self.path_length(tree.rtree, observation_id, length + 1)

    @staticmethod
    def average_path_length_in_bst(n: int) -> float:
        """Calculate the average path length of unsuccessful search in a BST of size `n`

        Args:
            n (int): the size of the BST

        Returns:
            float: the average path length
        """
        H = lambda n: log(n) + np.euler_gamma
        return 2 * H(n - 1) - (2 * (n - 1) / n)

    def grow_forest(self):
        """Grow the forest given the parameters set in init"""
        # no need of parallelization since the growth time is trivial
        for _ in range(self.forest_size):
            # sampling a set of observations by their indices from [0, len(self.obs))
            # 1st observation's index is 0
            # last observation's index is len(self.obs)-1
            # so `range(len(self.obs))` does include the last observation
            # `range()` here is used for sampling speed
            sample = random.sample(range(len(self.obs)), k=self.tree_size)
            self.trees.append(self.grow_tree(sample))

    def grow_tree(self, obs: list, current_height=0) -> IsolationTree:
        """Grow a single Isolation Tree

        Args:
            obs (list): list of observation indices
            current_height (int, optional): current tree height. Defaults to 0.

        Returns:
            IsolationTree: Isolation Tree grown
        """
        tree = IsolationTree(n_obs=len(obs))
        if len(obs) <= 1 or current_height >= self.max_tree_height:
            # when the search is terminated either case, the node is an ExNode
            tree.is_ex_node = True
            return tree
        # randomly choose an attribute `p` on which to split the sample
        # `p` is an integer from [0, len(self.attributes)) instead of the column name
        # because we are working with 2d array here
        p: int = random.choice(range(len(self.attributes)))
        # most basic way of finding min and max values by scanning only the necessary obs
        # TODO: have to deal with missing values
        max_val, min_val = -np.inf, np.inf
        for i in obs:
            max_val = max(max_val, self.data[p, i])
            min_val = min(min_val, self.data[p, i])

        # TODO: open question, from the uniform distribution or from the sample?
        # randomly choose a split value `q` from the uniform distribution [min_val, max_val]
        # q = random.uniform(min_val, max_val)
        q = random.choice([self.data[p][i] for i in obs])

        tree.split_attr, tree.split_val = p, q
        # recursively grow the left and right trees
        # TODO: improve efficiency here
        tree.ltree = self.grow_tree(
            [i for i in obs if self.data[p, i] <= q], current_height + 1
        )
        tree.rtree = self.grow_tree(
            [i for i in obs if self.data[p, i] > q], current_height + 1
        )
        return tree
