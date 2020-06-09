import os
import numpy as np
from .ra import ra
from .data import DataManager
from .measures import sample_measure
from multiprocessing.shared_memory import SharedMemory

if __name__ == "__main__":
    print('run!')

    with DataManager() as dm:
        shm, shape, dtype = dm.get_dataset(sample_measure.dataset)
        print(f'{shm=}, {shape=}, {dtype=}')
        nparray = np.recarray(shape=shape, dtype=dtype, buf=shm.buf)
        print(f'{nparray=}')
