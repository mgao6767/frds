import os
import json
import pandas as pd
from frds.data import CREDENTIALS_FILE_PATH
from frds.data.wrds import WRDSDataset
from frds.io.wrds.connection import Connection, CredentialsError


def setup(username="", password="", save_credentials=False):

    credentials = dict()
    if os.path.exists(CREDENTIALS_FILE_PATH):
        with open(CREDENTIALS_FILE_PATH) as fin:
            credentials = json.load(fin)

    usr = credentials.get("wrds_username", username)
    pwd = credentials.get("wrds_password", password)

    os.environ["frds_credentials_wrds_username"] = usr
    os.environ["frds_credentials_wrds_password"] = pwd

    if save_credentials:
        credentials = dict(wrds_username=usr, wrds_password=pwd)
        with open(CREDENTIALS_FILE_PATH, "w") as fout:
            json.dump(credentials, fout)


# Call setup on import so as to make available the credentials in env
setup(save_credentials=False)


def load(dataset: WRDSDataset, use_cache=True, save=True, obs=-1) -> WRDSDataset:

    if use_cache and os.path.exists(dataset.local_path):
        return dataset(pd.read_stata(dataset.local_path))

    usr = os.getenv("frds_credentials_wrds_username", "")
    pwd = os.getenv("frds_credentials_wrds_password", "")

    if usr == "" or pwd == "":
        raise CredentialsError

    with Connection(usr, pwd) as conn:
        tbl = conn.get_table(
            dataset.library,
            dataset.table,
            # index_col=dataset.index_col,
            date_cols=dataset.date_cols,
            obs=obs,
        )

    tmp = dataset(tbl)

    if save:
        tmp.data.to_stata(
            tmp.local_path,
            version=117,
            convert_strl=tmp.data.columns[
                tmp.data.isnull().all() & (tmp.data.dtypes == object)
            ].tolist(),
        )

    return tmp
