import os
import numpy as np
from .ra import RA
from .data import DataManager
from .measures import sample_measure
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pipe

if __name__ == "__main__":
    print('run!')

    measures = [sample_measure] * 10
    research_assistants = []

    with DataManager() as dm:
        # TODO: load data asynchronously
        for m in measures:
            shared_memory_info = dm.get_dataset(m.dataset)
            task = shared_memory_info, m.name, m.dataset, m.estimate
            (ra := RA(task, dm.queue)).start()
            research_assistants.append(ra)

        for ra in research_assistants:
            ra.join()

        while not dm.queue.empty():
            item = dm.queue.get()
            print(f'{item}')
