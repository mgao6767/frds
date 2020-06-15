import os
from .ra import RA
from .data import DataManager, Dataset
from .data.data_manager import SharedMemoryInfo
from frds.measures import *
from frds import result_dir, data_dir, wrds_password, wrds_username
from typing import List, Dict, Tuple, Set
from itertools import chain
import inspect
import frds


def main(measures=None, gui=False, config=None, progress_callback=None):
    if config is None:
        config = dict(
            wrds_username=wrds_username,
            wrds_password=wrds_password,
            result_dir=str(result_dir),
            data_dir=str(data_dir),
        )
    if gui:
        from importlib import import_module
        measures = [import_module(
            f'frds.measures.{measure}') for measure in measures]
        progress = progress_callback.emit
    else:
        measures = [measure for _, measure in inspect.getmembers(
            frds.measures, inspect.ismodule)]
        progress = print

    research_assistants = []

    # Consolidate data sources and make one request for one table
    progress('Checking datasets required for estimation...')
    datasets = set(dataset for m in measures for dataset in m.datasets)
    sources = set(dataset.source for dataset in datasets)
    progress(f'Data sources required: {",".join(sources)}.')
    progress('Preparing datasets...')
    tables_required = set((dta.source, dta.library, dta.table)
                          for m in measures for dta in m.datasets)
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
    progress('Creating data manager...')
    with DataManager(obs=-1, config=config) as dm:
        progress('Loading datasets...')
        shminfo: Dict[Dataset, SharedMemoryInfo] = dm.get_datasets(datasets)
        progress('Starting RAs...')
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
        progress('RAs are working on estimation...')
        for ra in research_assistants:
            ra.join()
        progress('Saving results...')
        os.makedirs(config.get('result_dir'), exist_ok=True)
        for item in dm.result:
            stata_file = f'{config.get("result_dir")}/{item["name"]}.dta'
            datasets: List[Dataset] = item['datasets']
            variable_labels: dict = item['variable_labels']
            date_vars = chain.from_iterable(
                dataset.date_vars for dataset in datasets)
            item['result'].to_stata(stata_file, write_index=False,
                                    convert_dates={v: 'td' for v in date_vars},
                                    variable_labels=variable_labels)
    progress(f'Completed! Results saved in {config.get("result_dir")}')


if __name__ == "__main__":
    main()
