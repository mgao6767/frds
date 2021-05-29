import os
import json
import pandas as pd
import sqlalchemy as sa
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


def load(
    dataset: WRDSDataset, columns=None, use_cache=True, save=True, obs=-1
) -> WRDSDataset:

    engine = sa.create_engine(f"sqlite:///{dataset.local_path}")

    if use_cache and os.path.exists(dataset.local_path):
        if obs == -1:
            return dataset(pd.read_sql_table(dataset.table, engine, columns=columns))
        else:
            return dataset(
                pd.read_sql_query(
                    f"SELECT * FROM {dataset.table} LIMIT {int(obs)};", engine
                )
            )

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
        tmp.data.to_sql(dataset.table, engine, if_exists="append", chunksize=1024)

    engine.dispose()
    return tmp
