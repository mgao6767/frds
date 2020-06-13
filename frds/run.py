from .ra import RA
from .data import DataManager, Dataset
from .data.data_manager import SharedMemoryInfo
from .measures import tangibility, book_leverage, roa, roe
from frds import result_dir
from typing import List, Dict, Tuple, Set
from itertools import chain

if __name__ == "__main__":
    print('run!')

    measures = [roa, tangibility, roe, book_leverage]
    research_assistants = []

    # Consolidate data sources and make one request for one table
    print('Checking datasets required for estimation...')
    datasets = set((dataset for m in measures for dataset in m.datasets))
    sources = set([dataset.source for dataset in datasets])
    print(f'Data sources required: {",".join(sources)}.')
    print(f'Preparing datasets...')
    tables_required = set([(dta.source, dta.library, dta.table)
                           for m in measures for dta in m.datasets])
    table_vars: Dict[Tuple, Set] = {table: set() for table in tables_required}
    table_dates: Dict[Tuple, Set] = {table: set() for table in tables_required}
    for table in tables_required:
        for dta in datasets:
            if (dta.source, dta.library, dta.table) == table:
                table_vars.get(table).update(dta.vars)
                table_dates.get(table).update(dta.date_vars)
    datasets = []
    for table in tables_required:
        src, lib, tablename = table
        datasets.append(Dataset(source=src,
                                library=lib,
                                table=tablename,
                                vars=table_vars.get(table),
                                date_vars=table_dates.get(table)))

    with DataManager(obs=-1) as dm:

        shminfo: Dict[Dataset, SharedMemoryInfo] = dm.get_datasets(datasets)

        for m in measures:
            # find the required tables for this measure m
            tables_required = [(dta.source, dta.library, dta.table)
                               for dta in m.datasets]
            # find the related dataset in shared memory
            datasets_related = {dta: shm for dta, shm in shminfo.items() if
                                (dta.source, dta.library, dta.table) in tables_required}
            task = datasets_related, m.name, m.datasets, m.estimate
            (ra := RA(task, dm.result)).start()
            research_assistants.append(ra)

        for ra in research_assistants:
            ra.join()

        for item in dm.result:
            stata_file = f'{result_dir}/{item["name"]}.dta'
            datasets: List[Dataset] = item['datasets']
            variable_labels: dict = item['variable_labels']
            date_vars = chain.from_iterable(
                dataset.date_vars for dataset in datasets)
            item['result'].to_stata(stata_file, write_index=False,
                                    convert_dates={v: 'td' for v in date_vars},
                                    variable_labels=variable_labels)
