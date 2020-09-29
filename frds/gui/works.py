import functools
import pathlib
import os
from frds.gui.multiprocessing import ThreadWorker
from frds.data import wrds
from frds.utils.settings import read_data_source_credentials, read_general_settings


@functools.lru_cache()
def get_wrds_connection():
    credentials = read_data_source_credentials()
    username = credentials.get("wrds_username")
    password = credentials.get("wrds_password")
    return wrds.Connection(username, password)


def make_worker_list_wrds_tables(library) -> ThreadWorker:
    """Return a ThreadWorker to list the available tables in a library"""

    def list_wrds_tables(library):
        conn = get_wrds_connection()
        return library, conn.list_tables(library)

    return ThreadWorker(f"List tables of {library}", list_wrds_tables, library)


def make_worker_download_wrds_table(library, table) -> ThreadWorker:
    """Return a ThreadWorker to download the dataset"""

    def download(library, table):
        conn = get_wrds_connection()
        return conn.get_table(library, table, obs=100)

    return ThreadWorker(f"Download {library}.{table}", download, library, table)


def make_worker_save_dataset(source, library, table, df) -> ThreadWorker:
    """Return a ThreadWorker to save the dataset to destination path"""

    def save(source, library, table, df):
        settings = read_general_settings()
        data_dir = settings.get("data_dir")
        dataset_dir = pathlib.Path(data_dir).joinpath(source, library).expanduser()
        os.makedirs(dataset_dir.as_posix(), exist_ok=True)
        df.to_csv(dataset_dir.joinpath(f"{table}.csv").as_posix())

    return ThreadWorker(
        f"Save {source}.{library}.{table}", save, source, library, table, df,
    )


def make_worker_download_and_save_wrds_table(source, library, table) -> ThreadWorker:
    """Return a ThreadWorker to download the dataset"""

    def download_and_save(library, table):
        settings = read_general_settings()
        data_dir = settings.get("data_dir")
        dataset_dir = pathlib.Path(data_dir).joinpath(source, library).expanduser()
        os.makedirs(dataset_dir.as_posix(), exist_ok=True)
        filepath = dataset_dir.joinpath(f"{table}.csv")
        if filepath.exists():
            return
        conn = get_wrds_connection()
        df = conn.get_table(library, table, obs=100)
        df.to_csv(filepath.as_posix(), index=False)

    return ThreadWorker(
        f"Download & Save {source}.{library}.{table}", download_and_save, library, table
    )


def make_worker_list_wrds_libraries() -> ThreadWorker:
    """Return a ThreadWorker to list the WRDS libraries"""

    def list_wrds_libraries():
        conn = get_wrds_connection()
        return conn.list_libraries()

    return ThreadWorker("List WRDS libraries", list_wrds_libraries)

