from numpy import recarray
from multiprocessing import Process
from ..data import DataManager


class RA(Process):

    def __init__(self, task, result_queue):
        super(RA, self).__init__()
        self.result_queue = result_queue
        lst_shared_memory_info, self.name, self.dataset, self.func = task
        self.nparrays = [
            recarray(shape=shape, dtype=dtype, buf=shm.buf)
            for shm, shape, dtype in lst_shared_memory_info
        ]

    def run(self):
        self.result_queue.put(
            dict(name=self.name, dataset=self.dataset,
                 result=self.func(self.nparrays)))
