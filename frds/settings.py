"""General settings of FRDS"""

import os

FRDS_HOME_PAGE = "https://frds.io"

ROOT_PATH = "~/frds"
CONFIG_FILE_PATH = "~/frds"
CONFIG_FILE_NAME = "config.ini"

MAX_WORKERS = os.cpu_count()
PROGRESS_UPDATE_INTERVAL_SECONDS = 1
