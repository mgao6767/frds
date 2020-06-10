from numpy import recarray
from multiprocessing import Process
from ..data import DataManager


class RA(Process):

    def __init__(self, task, result):
        super(RA, self).__init__()
        self.result = result
        lst_shared_memory_info, self.name, self.datasets, self.func = task
        self.nparrays = [
            recarray(shape=shape, dtype=dtype, buf=shm.buf)
            for shm, shape, dtype in lst_shared_memory_info
        ]

    def run(self):
        self.result.append(
            dict(name=self.name, datasets=self.datasets,
                 result=self.func(self.nparrays)))
