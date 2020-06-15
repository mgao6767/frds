from numpy import recarray
from multiprocessing import Process
from ..data import DataManager, Dataset
from itertools import chain


def table_of(dataset: Dataset):
    return dataset.source, dataset.library, dataset.table


class RA(Process):

    def __init__(self, task, config: dict):
        super(RA, self).__init__()
        self.config = config
        datasets_related, self.name, self.datasets, self.func = task
        # must keep the order of required datasets
        self.nparrays = []
        for dataset in self.datasets:
            for dta, (shm, shape, dtype) in datasets_related.items():
                if table_of(dta) == table_of(dataset):
                    self.nparrays.append(
                        recarray(shape=shape, dtype=dtype, buf=shm.buf))

    def run(self):
        result, variable_labels = self.func(self.nparrays)
        # RAs can save results themselves. No need to return to main process
        stata_file = f'{self.config.get("result_dir")}/{self.name}.dta'
        date_vars = chain.from_iterable(
            dataset.date_vars for dataset in self.datasets)
        result.to_stata(stata_file, write_index=False,
                        convert_dates={
                            v: 'td' for v in date_vars if v in result.columns},
                        variable_labels=variable_labels)
