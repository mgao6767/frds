from numpy import recarray
from multiprocessing import Process
from ..data import DataManager, Dataset


def table_of(dataset: Dataset):
    return dataset.source, dataset.library, dataset.table


class RA(Process):

    def __init__(self, task, result):
        super(RA, self).__init__()
        self.result = result
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
        self.result.append(
            dict(name=self.name, datasets=self.datasets,
                 result=result, variable_labels=variable_labels))
