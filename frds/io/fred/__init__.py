import os
import json
from frds.data import CREDENTIALS_FILE_PATH

from fredapi import Fred


# TODO: Update logic here. Now, credentials.json won't be updated if new ones are supplied.
def setup(api_key="", save_credentials=False):

    credentials = dict()
    if os.path.exists(CREDENTIALS_FILE_PATH):
        with open(CREDENTIALS_FILE_PATH) as fin:
            credentials = json.load(fin)

    key = credentials.get("fred_api_key", api_key)

    os.environ["frds_credentials_fred_api_key"] = key

    if save_credentials:
        # credentials.update({"fred_api_key": key})
        credentials["fred_api_key"] = key
        with open(CREDENTIALS_FILE_PATH, "w") as fout:
            json.dump(credentials, fout)
        print(key)


# Call setup on import so as to make available the credentials in env
setup()

fred = Fred(api_key=os.getenv("frds_credentials_fred_api_key", ""))
