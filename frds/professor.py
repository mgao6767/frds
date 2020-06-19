"""Define the professor who's the main supervisor (process) of estimations"""

import os
from typing import List, Set, Dict
from multiprocessing import Manager, active_children
from frds.ra import ResearchAssistant
from frds.measures import Measure
from frds.data import Dataset, DataManager
from frds.typing import TableID, SharedMemoryInfo


class Professor:
    """Professor is the main supervisor (process) for the entire estimation.
    The professor dispatches jobs to `Research Assistant`s who actually perform
    the estimation.

    Parameters
    ----------
    assistants : int, optional
        The maximum concurrency, by default os.cpu_count(). Note that because
        some measures may spwan subprocesses as well. This doesn't guarantee the
        actual number of subprocesses will be no more than `assistants`.
    """

    def __init__(self, assistants=os.cpu_count(), config=None, progress=None):
        self._assistants = assistants
        self._datamanager = DataManager(config=config)
        # The professor's office is where communication happens!
        self.office = Manager().Namespace()
        # The database stores the shared memory info for each table
        self.office.database: Dict[TableID, SharedMemoryInfo] = None
        # The configuration of login info and data/result directories
        self.office.config = config
        # Function to update progress message
        self.progress = print if progress is None else progress

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._datamanager.shutdown()

    @staticmethod
    def _prepare_datasets(measures: List[Measure]) -> List[Dataset]:
        """Given a list of measures, return the consolidated datasets required

        This is necessary because some measures may require the same dataset,
        although different variables. So the professor is going to consolidate
        the required datasets and make a bulk request to the data manager.

        Parameters
        ----------
        measures : List[Measure]
            List of measures to estimate

        Returns
        -------
        List[Dataset]
            List of consolidated datasets required
        """
        # Get a se of distinct table ids
        table_ids: Set[TableID] = set(
            dta.table_id for m in measures for dta in m.datasets_required
        )
        # Init empty dicts: TableID -> Set[VarName]
        table_varnames = {table_id: set() for table_id in table_ids}
        table_datevars = {table_id: set() for table_id in table_ids}
        # Find for each table, the distinct variables required for all measures
        for dta in (dta for m in measures for dta in m.datasets_required):
            table_varnames.get(dta.table_id).update(dta.vars)
            table_datevars.get(dta.table_id).update(dta.date_vars)
        # Consolidate datasets
        datasets: List[Dataset] = []
        for table_id in table_ids:
            src, lib, table = table_id
            varnames = list(table_varnames.get(table_id))
            datevars = list(table_datevars.get(table_id))
            datasets.append(Dataset(src, lib, table, varnames, datevars))
        return datasets

    @staticmethod
    def _prioritse_measures(measures: List[Measure]) -> List[Measure]:
        """Prioritise the measures to estimate
        
        Measures using the same dataset should be clustered and scheduled to run
        together if possible. Datasets no longer needed can then be removed.

        Parameters
        ----------
        measures : List[Measure]
            List of measures to estimate

        Returns
        -------
        List[Measure]
            List of measures to estimate (potentially reordered)
        """
        # TODO: to be implemented
        return measures

    def calculate(self, measures: List[Measure]) -> None:
        """The professor is not going to caculate the measures but will allocate
        the tasks to research assistants.

        Parameters
        ----------
        measures : List[Measure]
            Measures need to estimate
        """
        self.progress("Preparing datasets...")
        datasets = self._prepare_datasets(measures)

        self.progress("Prioritse measures to estimate...")
        measures = self._prioritse_measures(measures)

        self.progress("Requesting data from the data manager...")
        with self._datamanager as data_manager:
            self.office.data = data_manager.get_datasets(datasets)
            # Professor doesn't need to to tell RAs which datasets are required
            # for each measure, because RAs can figure out themselves!
            self.progress("Starting allocating jobs to research assistant...")
            assistants = []
            while True:
                # Limit concurrency
                if len(active_children()) < self._assistants:
                    if not measures:
                        break
                    measure = measures.pop(0)
                    assistant = ResearchAssistant(measure, self.office)
                    self.progress(f"Starting RA for {measure.name}...")
                    assistant.start()
                    assistants.append(assistant)
            self.progress("Waiting for RAs to finish estimation...")
            for assistant in assistants:
                assistant.join()
        result_dir = self.office.config.get("result_dir")
        self.progress(f"Completed! Results saved in {result_dir}")
