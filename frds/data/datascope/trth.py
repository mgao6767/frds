import requests
import zipfile
import tempfile
import json
import time
import gzip
import os

import pandas as pd

from .utils import make_request_index_components
from .request_templates import INTRADAY_TICKS

URL_BASE = "https://hosted.datascopeapi.reuters.com/RestApi/v1"

AUTH_URL = f"{URL_BASE}/Authentication/RequestToken"
SEARCH_HIST_CHAIN_URL = f"{URL_BASE}/Search/HistoricalChainResolution"
EXTRACT_RAW_URL = f"{URL_BASE}/Extractions/ExtractRaw"
RESULTS_URL = f"{URL_BASE}/Extractions/RawExtractionResults('<JobId>')/$value"


class Connection:
    def __init__(self, usr, pwd, token=None, queryString=None, *args, **kwargs):
        self._username = usr
        self._password = pwd
        self.queryString = queryString
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Prefer": "respond-async",
                "Content-Type": "application/json",
                "Accept-Charset": "UTF-8",
            }
        )
        self.session.hooks["response"] = [
            self._printRequestURL,
            self._checkResponseForError,
        ]
        self._accessToken = self._getAccessToken() if token is None else token
        self.updateAccessTokenInRequestHeaders(self._accessToken)
        self.pollingIntervalSeconds = 10

    def close(self):
        pass

    def get_index_components(self, mkt_index, date_start, date_end):
        payload = make_request_index_components(mkt_index, date_start, date_end)
        resp = self.session.post(SEARCH_HIST_CHAIN_URL, payload)
        return resp.json()

    def get_table(self, library, table, columns, date_cols, obs):
        if library.lower() == "datascope" and table.lower() == "trth":
            return self.extract_raw(INTRADAY_TICKS)

    def _getAccessToken(self) -> str:
        """Return the access Token"""
        _data = {
            "Credentials": {
                "Username": self._username,
                "Password": self._password,
            }
        }
        resp = self.session.post(AUTH_URL, json=_data)
        return resp.json().get("value", "")

    def updateAccessTokenInRequestHeaders(self, token: str) -> None:
        self.session.headers.update({"Authorization": f"Token {token}"})

    @staticmethod
    def _printRequestURL(resp, *args, **kwargs):
        """Hook function to print the request URL"""
        print(f"Request url: {resp.url} Status: {resp.status_code}")

    # @staticmethod
    def _checkResponseForError(self, resp, *args, **kwargs):
        """Hook function to be called after every response"""
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            # The tokens are valid for 24 hours, after which
            # a 401 (Unauthorized, Authentication Required) status code is returned.
            if resp.status_code == 401:
                _newToken = self._getAccessToken()  # Get a new token
                self.updateAccessTokenInRequestHeaders(
                    _newToken
                )  # Update token for the session
                self.session.send(resp.request)  # Resend the request
            # Raise the error if it's not due to invalid token.
            if resp.status_code == 400:
                print(f"Error: {resp.text}")

    def extract_raw(self, payload: dict):
        resp = self.session.post(EXTRACT_RAW_URL, json=payload)
        _location = resp.headers.get("location").replace("http://", "https://")
        while resp.status_code != 200:
            time.sleep(self.pollingIntervalSeconds)
            resp = self.session.get(_location)
        print(f"Location: {_location}")
        resp_json = resp.json()
        job_id = resp_json.get("JobId")
        print(f"Job ID is {job_id}")
        # Check if the response contains Notes.If the note exists print it to console.
        if len(resp_json.get("Notes")) > 0:
            print("Notes:\n======================================")
            for var in resp_json.get("Notes"):
                print(var)
            print("======================================\n")
        # Request should be completed then Get the result by passing jobID to RAWExtractionResults URL
        resultURL = RESULTS_URL.replace("<JobId>", job_id)
        print(f"Retrieve result from {resultURL}")
        # Allow downloading directly from AWS
        resp = self.session.get(
            resultURL, stream=True, headers={"X-Direct-Download": "true"}
        )
        _output_file = tempfile.NamedTemporaryFile()
        # Write Output to file
        for chunk in resp.iter_content(chunk_size=1024):
            _output_file.write(chunk)
        _output_file.seek(0)
        df = pd.read_csv(_output_file, compression="gzip")
        _output_file.close()
        return df
