from numpy import recarray
from multiprocessing import Process
from ..data import DataManager


class RA(Process):

    def __init__(self, task, result_queue):
        super(RA, self).__init__()
        shared_memory_info, self.name, self.dataset, self.func = task
        shm, shape, dtype = shared_memory_info
        self.nparray = recarray(shape=shape, dtype=dtype, buf=shm.buf)
        self.result_queue = result_queue

    def run(self):
        self.result_queue.put(
            dict(name=self.name, dataset=self.dataset,
                 result=self.func(self.nparray)))
