import pandas as pd
from ..algorithms.isolation_forest import anomaly_scores


def anomaly_score(
    data: pd.DataFrame,
    forest_size: int = 1000,
    tree_size: int = 256,
    exclude_cols: list = None,
    random_seed: int = 1,
    name: str = "AnomalyScore",
) -> pd.DataFrame:
    """Calculate the anomaly socres using Isolation Forest

    Args:
        data (pd.DataFrame): indexed DataFrame
        tree_size (int, optional): number of observations per Isolation Tree. Defaults to 256.
        forest_size (int, optional): number of trees. Defaults to 1_000.
        exclude_cols (list, optional): columns in the input `data` to ignore. Defaults to None.
        random_seed (int, optional): random seed for reproducibility. Defaults to 1.
        name (str, optional): column name of the return DataFrame. Defaults to "AnomalyScore".

    Returns:
        pd.DataFrame: single-column indexed DataFrame

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from frds.measures import anomaly_score
        >>> n_obs, n_attrs = 1_000, 100
        >>> np.random.seed(0)
        >>> data = pd.DataFrame(
        ...     {f"Attr{i}": np.random.normal(0, 1, n_obs) for i in range(n_attrs)},
        ...     index=[f"obs.{i}" for i in range(n_obs)],
        ... )

        Create an outlier
        >>> data.loc["obs.o"] = 10
        >>> data.head()
                Attr0      Attr1      Attr2      Attr3      Attr4      Attr5  ...     Attr94     Attr95     Attr96     Attr97     Attr98     Attr99
        obs.0  10.000000  10.000000  10.000000  10.000000  10.000000  10.000000  ...  10.000000  10.000000  10.000000  10.000000  10.000000  10.000000
        obs.1   0.400157   0.892474  -1.711970   0.568722   1.843700  -0.737456  ...  -0.293003   1.439150  -1.912125  -0.041582  -0.359808  -0.910595
        obs.2   0.978738  -0.422315   0.046135  -0.114487   0.271091  -1.536920  ...   1.873511  -1.500278  -1.073391   0.462102   1.425187   0.187550
        obs.3   2.240893   0.104714  -0.958374   0.251630   1.136448  -0.562255  ...   2.274601   0.851165   0.774576   0.087572  -0.013497  -0.514234
        obs.4   1.867558   0.228053  -0.080812  -1.210856  -1.738332  -1.599511  ...   0.821944   0.512349  -0.263450  -0.583916  -0.972011  -0.527498
        [5 rows x 100 columns]

        Check anomaly scores
        >>> anomaly_score(data)
                AnomalyScore
        obs.0        0.681283
        obs.1        0.511212
        obs.2        0.500675
        obs.3        0.496884
        obs.4        0.505786
        ...               ...
        obs.995      0.486229
        obs.996      0.488048
        obs.997      0.479535
        obs.998      0.493547
        obs.999      0.498489

        Verify that obs.0 has the highest anomaly score
        >>> anomaly_score(data).sort_values(by="AnomalyScore", ascending=False).head()
                AnomalyScore
        obs.0        0.681283
        obs.766      0.527717
        obs.911      0.525643
        obs.676      0.525509
        obs.71       0.525022
    """
    return anomaly_scores(data, forest_size, tree_size, exclude_cols, random_seed, name)
