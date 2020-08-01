"""DataManager manages the shared memory"""

import os
from typing import List, Dict
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from importlib import import_module
import numpy as np
import pandas as pd
from frds.typing import TableID, SharedMemoryInfo
from .dataset import Dataset


class Singleton(object):
    """Singleton implementation by Guido van Rossum"""

    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        pass


class DataManager(Singleton):
    """DataManager loads data from sources and manages the shared memory"""

    def __init__(self, obs=-1, config=None):
        self._datasets = dict()
        self.smm = SharedMemoryManager()
        self.smm.start()
        self.conns = dict()
        self.obs = obs
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        """Shutdown the shared memory manager and all data connections"""
        self.smm.shutdown()
        for conn in self.conns.values():
            conn.close()

    def _add_to_shared_memory(self, nparray: np.recarray) -> SharedMemory:
        """Internal function to copy an array into shared memory.

        Parameters
        ----------
        nparray : np.recarray
            The array to be copied into shared memory.

        Returns
        -------
        SharedMemory
            The shared memory object.
        """
        shm = self.smm.SharedMemory(nparray.nbytes)
        array = np.recarray(nparray.shape, dtype=nparray.dtype, buf=shm.buf)
        np.copyto(array, nparray)
        return shm

    def _download_dataset(self, dataset: Dataset) -> pd.DataFrame:
        """Internal function to download the dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset information including source, library, table, vars, etc.

        Returns
        -------
        pd.DataFrame
            DataFrame of the downloaded dataset.
        """
        # TODO: generic login data for different data sources
        if dataset.source == "wrds":
            usr = self.config.get("wrds_username")
            pwd = self.config.get("wrds_password")
        if dataset.source == "frb_chicago":
            usr, pwd = None, None
        # If there exists a connection for the data source, use it!
        if (conn := self.conns.get(dataset.source, None)) is None:
            module = import_module(f"frds.data.{dataset.source}")
            conn = module.Connection(usr=usr, pwd=pwd)
            self.conns.update({dataset.source: conn})
        return conn.get_table(
            library=dataset.library,
            table=dataset.table,
            columns=dataset.vars,
            date_cols=dataset.date_vars,
            obs=self.obs,
        )

    def _get_dataset(self, dataset: Dataset) -> SharedMemoryInfo:
        """Get the shared memory object, shape and dtype of the given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset information including source, library, table, vars, etc.

        Returns
        -------
        SharedMemoryInfo
            (SharedMemory, tuple, numpy.dtype), where the tuple is the shape.
        """
        if dataset in self._datasets:
            return self._datasets.get(dataset)
        dataset_name = f"{dataset.source}_{dataset.library}_{dataset.table}"
        path = os.path.join(self.config.get("data_dir"), f"{dataset_name}.dta")
        # If local data directory has the table
        if os.path.exists(path):
            df = next(pd.read_stata(path, chunksize=1))
            # If required variables are all in the table
            if set(dataset.vars).issubset(set(df.columns)):
                df = pd.read_stata(path, columns=dataset.vars)
            # Some variables are missing and need to be downloaded
            else:
                # Should find out the missing variables and only download them
                # But since we don't know the identifiers, might as well just
                # download everything and merge with existing one.
                df = self._download_dataset(dataset)
                df_in_datadir = pd.read_stata(path)
                common_cols = [
                    col for col in df.columns if col in df_in_datadir.columns
                ]
                df_to_save = df.merge(df_in_datadir, on=common_cols)
                df_to_save.to_stata(
                    path,
                    write_index=False,
                    convert_dates={v: "td" for v in dataset.date_vars},
                )
                del df_in_datadir
                del df_to_save
        # No such table in the local data directory
        else:
            # Download the entire table from source
            df = self._download_dataset(dataset)
            # Store the table in local data directory
            df.to_stata(
                path,
                write_index=False,
                convert_dates={v: "td" for v in dataset.date_vars},
            )
        # VERY IMPORTANT: must find out the string-typed columns
        dtypes = df.convert_dtypes().dtypes
        str_cols = dtypes.where(dtypes == "string").dropna().index.to_list()
        # Stata variable name's length is maxed at 32. Here I use 32 characters
        # unicode dtype because "S32" maps to zero-terminated bytes `np.bytes_`
        # for backward compatibility in Python 2. In Python 3 we use unicode.
        # This will slightly increase the array size, but should be no harm.
        nparray = df.to_records(
            index=False, column_dtypes={col: "U32" for col in str_cols}
        )
        shm = self._add_to_shared_memory(nparray)
        self._datasets.update({dataset: (shm, nparray.shape, nparray.dtype)})
        return shm, nparray.shape, nparray.dtype

    def get_datasets(
        self, datasets: List[Dataset]
    ) -> Dict[TableID, SharedMemoryInfo]:
        """Return a dictionary of {TableID: SharedMemoryInfo}.

        Parameters
        ----------
        datasets : List[Dataset]
            List of requried datasets.

        Returns
        -------
        Dict[TableID, SharedMemoryInfo]
            Dictionary of TableID to SharedMemoryInfo.
        """
        return {dta.table_id: self._get_dataset(dta) for dta in datasets}

