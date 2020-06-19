"""Defines the base class for all measures"""
import abc
from typing import List, Tuple
import numpy as np
import pandas as pd
from frds.data import Dataset
from frds.typing import TableID


class Measure(abc.ABC):
    """Base class for all measures"""

    def __init__(self, name: str, datasets_required: List[Dataset]):
        self._name = name
        self._datasets_required = datasets_required

    @property
    def name(self) -> str:
        """Return the measure's name

        Returns
        -------
        str
            Name of the measure
        """
        return self._name

    def __str__(self) -> str:
        """Return the measure's name by default. However, actual measures can
        choose to overwrite this if they have custom parameters necessary to
        identify the measure

        Returns
        -------
        str
            Name of the measure
        """
        return self._name

    @property
    def datasets_required(self) -> List[Dataset]:
        """Return the measure's required datasets

        Returns
        -------
        List[Dataset]
            List of required `frds.data.Dataset`s
        """
        return self._datasets_required

    @property
    def data_sources(self) -> List[str]:
        """Return the list of unique data sources required

        Returns
        -------
        List[str]
            List of required data sources
        """
        return list(set(dta.source for dta in self._datasets_required))

    @property
    def data_tables(self) -> List[TableID]:
        """Return the list of unique tables required

        Returns
        -------
        list[TableID]
            List of (dataset.source, dataset.library, dataset.table)
        """
        return list(set(dta.table_id for dta in self._datasets_required))

    @abc.abstractmethod
    def estimate(
        self, nparrays: List[np.recarray]
    ) -> Tuple[pd.DataFrame, dict]:
        """Estimatation function for the measure to be implemented

        Parameters
        ----------
        nparrays : List[np.recarray]
            List of `numpy.recarray`s corresponding to the required datasets

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            Tuple of result `pandas.DataFrame` and dictionay of variable labels
        """
        raise NotImplementedError
