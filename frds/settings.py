"""General settings of FRDS"""

import os

FRDS_HOME_PAGE = "https://frds.io"

ROOT_PATH = "~/frds"
CONFIG_FILE_PATH = "~/frds"
CONFIG_FILE_NAME = "config.ini"

ABOUT_FRDS = """
FRDS is written by [Mingze Gao](https://mingze-gao.com) from the University of Sydney. \
It is used to compute a collection of academic measures in corporate finance, banking, \
and market microstructure researches.

If you encounter any bug or issue when using FRDS, please contact me at \
[mingze.gao@sydney.edu.au](mingze.gao@sydney.edu.au). You can also raise an issue on \
[GitHub](https://github.com/mgao6767/frds/issues/new).

This software is provided free-of-charge under the MIT license. \
The graphical user interface (GUI) is build with PyQt5 (GNU license).
"""

MAX_WORKERS = os.cpu_count()
PROGRESS_UPDATE_INTERVAL_SECONDS = 1
