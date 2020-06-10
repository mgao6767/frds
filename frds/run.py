from .ra import RA
from .data import DataManager, Dataset
from .measures import tangibility
from frds import result_dir
from typing import List
from itertools import chain

if __name__ == "__main__":
    print('run!')

    measures = [tangibility]
    research_assistants = []

    with DataManager(obs=-1) as dm:
        # TODO: load data asynchronously
        for m in measures:
            lst_shared_memory_info = dm.get_datasets(m.datasets)
            task = lst_shared_memory_info, m.name, m.datasets, m.estimate
            (ra := RA(task, dm.result)).start()
            research_assistants.append(ra)

        for ra in research_assistants:
            ra.join()

        for item in dm.result:
            stata_file = f'{result_dir}/{item["name"]}.dta'
            datasets: List[Dataset] = item['datasets']
            date_vars = chain.from_iterable(
                dataset.date_vars for dataset in datasets)
            item['result'].to_stata(stata_file, write_index=False,
                                    convert_dates={v: 'td' for v in date_vars})
