"""Utility functions related to reading/saving overal program settings"""

import functools
import pathlib
import frds.settings


@functools.lru_cache()
def get_root_dir() -> pathlib.Path:
    path = pathlib.Path(frds.settings.ROOT_PATH).expanduser()
    return path.as_posix()
