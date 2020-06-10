from .ra import RA
from .data import DataManager
from .measures import sample_measure
from frds import result_dir

if __name__ == "__main__":
    print('run!')

    measures = [sample_measure]
    research_assistants = []

    with DataManager() as dm:
        # TODO: load data asynchronously
        for m in measures:
            lst_shared_memory_info = dm.get_datasets(m.datasets)
            task = lst_shared_memory_info, m.name, m.datasets, m.estimate
            (ra := RA(task, dm.queue)).start()
            research_assistants.append(ra)

        for ra in research_assistants:
            ra.join()

        while not dm.queue.empty():
            item = dm.queue.get()
            item['result'].to_stata(f'{result_dir}/{item["name"]}.dta')
