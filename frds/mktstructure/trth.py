import requests
import time

from .utils import make_request_index_components, make_request_tick_history

# URL_BASE = "https://hosted.datascopeapi.reuters.com/RestApi/v1"
URL_BASE = "https://selectapi.datascope.refinitiv.com/RestApi/v1"

AUTH_URL = f"{URL_BASE}/Authentication/RequestToken"
SEARCH_HIST_CHAIN_URL = f"{URL_BASE}/Search/HistoricalChainResolution"
EXTRACT_RAW_URL = f"{URL_BASE}/Extractions/ExtractRaw"
RESULTS_URL = f"{URL_BASE}/Extractions/RawExtractionResults('<JobId>')/$value"


class Connection:
    def __init__(self, usr, pwd, token=None, progress_callback=print, *args, **kwargs):
        self.print_fn = progress_callback
        self._username = usr
        self._password = pwd
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

    def get_table(self, rics, start_date, end_date):
        req = make_request_tick_history(rics, start_date, end_date)
        return self.extract_raw(req)

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
        self.session.headers.update({"Authorization": f"token {token}"})

    def _printRequestURL(self, resp, *args, **kwargs):
        """Hook function to self.print_fn the request URL"""
        self.print_fn(f"Request url: {resp.url} Status: {resp.status_code}")

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
                self.print_fn(f"Error: {resp.text}")

    def extract_raw(self, payload: dict):
        resp = self.session.post(EXTRACT_RAW_URL, json=payload)
        _location = resp.headers.get("location").replace("http://", "https://")
        while resp.status_code != 200:
            self.print_fn(
                f"Waiting for data delivery... Polling in {self.pollingIntervalSeconds}s."
            )
            time.sleep(self.pollingIntervalSeconds)
            resp = self.session.get(_location)
        # self.print_fn(f"Location: {_location}")
        resp_json = resp.json()
        job_id = resp_json.get("JobId")
        # self.print_fn(f"Job ID is {job_id}")
        # Check if the response contains Notes.If the note exists self.print_fn it to console.
        if len(resp_json.get("Notes")) > 0:
            for var in resp_json.get("Notes"):
                if "Quota" in var:
                    for line in var.split(";")[-3:]:
                        self.print_fn(line)
        # Request should be completed then Get the result by passing jobID to RAWExtractionResults URL
        resultURL = RESULTS_URL.replace("<JobId>", job_id)
        # self.print_fn(f"Retrieve result from {resultURL}")
        # Allow downloading directly from AWS
        resp = self.session.get(
            resultURL, stream=True, headers={"X-Direct-Download": "true"}
        )
        return resp

    @staticmethod
    def save_results(resp, path):
        # _output_file = tempfile.NamedTemporaryFile()
        # Write Output to file
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        # _output_file.seek(0)
        # df = pd.read_csv(_output_file, compression="gzip")
        # _output_file.close()
        # return df
