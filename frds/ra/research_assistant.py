"""Define the research assistant process"""

import os
from multiprocessing import Process
from multiprocessing.managers import Namespace
from typing import Dict
from numpy import recarray
from pandas import DataFrame
from frds.measures import Measure
from frds.typing import TableID, SharedMemoryInfo


class ResearchAssistant(Process):
    """Research Assistant is the worker (process) who manages and performs the
    estimation, as well as saving results.

    Parameters
    ----------
    measure : Measure
        The measure to estimate.
    
    data : Dict[TableID, SharedMemoryInfo]
        Used to know where to locate the required datasets in memory
    """

    def __init__(self, measure: Measure, office: Namespace):
        super(ResearchAssistant, self).__init__()
        self.measure = measure
        self.data: Dict[TableID, SharedMemoryInfo] = office.data
        self.config = office.config

    def run(self):
        # Load data from shared memory
        datasets = []
        for dta in self.measure.datasets_required:
            shm, shape, dtype = self.data.get(dta.table_id)
            datasets.append(recarray(shape, dtype=dtype, buf=shm.buf))
        # Invoke the estimation function of the measure
        result, variable_labels = self.measure.estimate(datasets)
        self._save_results(result, variable_labels)

    def _save_results(self, result: DataFrame, variable_labels: Dict[str, str]):
        """Save results to local result directory

        RAs can save results themselves. No need to return to main process

        Parameters
        ----------
        result : DataFrame
            Result dataset
        variable_labels : Dict[str, str]
            Dictionary of variable name to label
        """
        assert isinstance(result, DataFrame)
        # Find out the date variables
        date_vars = set()
        for dta in self.measure.datasets_required:
            date_vars.update(dta.date_vars)
        # Save to the result directory
        result_dir = self.config.get("result_dir")
        stata_file = f"{self.measure}.dta"
        stata_path = os.path.join(result_dir, stata_file)
        result.to_stata(
            path=stata_path,
            write_index=False,
            convert_dates={v: "td" for v in date_vars if v in result.columns},
            variable_labels=variable_labels,
        )

