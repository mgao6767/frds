import functools
from frds.gui.multiprocessing import ThreadWorker
from frds.data import wrds
from frds.utils.settings import read_data_source_credentials


@functools.lru_cache()
def get_wrds_connection():
    credentials = read_data_source_credentials()
    username = credentials.get("wrds_username")
    password = credentials.get("wrds_password")
    return wrds.Connection(username, password)


def get_wrds_libraries():
    conn = get_wrds_connection()
    return conn.list_libraries()


def get_wrds_tables(library):
    conn = get_wrds_connection()
    return conn.list_tables(library)


worker_list_wrds_libaries = ThreadWorker("Load WRDS libraries", get_wrds_libraries)
get_worker_list_wrds_tables = lambda library: ThreadWorker(
    f"{library}", get_wrds_tables, library
)
