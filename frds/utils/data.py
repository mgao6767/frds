import typing
import os
import pathlib
from datetime import datetime
import pandas as pd
from frds.utils.settings import read_general_settings


def get_market_microstructure_data(
    securities: typing.List[str], start_date: datetime, end_date: datetime
) -> typing.Iterator[typing.Tuple[datetime, str, pd.DataFrame]]:
    """Yield (date, security, data)"""

    settings = read_general_settings()
    data_dir = (
        pathlib.Path(settings.get("data_dir")).joinpath("TRTH").joinpath("parsed_data")
    ).expanduser()

    for date in pd.date_range(start_date, end_date):
        for security in securities:
            data = _load_data_for_security_date(security, date, data_dir)
            yield date, security, data
            del data


def get_total_files_of_market_microstructure_data_on_disk():
    """Return the number of parsed data files on disk"""
    settings = read_general_settings()
    data_dir = (
        pathlib.Path(settings.get("data_dir")).joinpath("TRTH").joinpath("parsed_data")
    ).expanduser()
    files = []
    for _, _, filenames in os.walk(data_dir.as_posix()):
        files.extend(filenames)
    return len(files)


def get_market_microstructure_data_from_disk(
    start_date: typing.Optional[datetime] = None,
    end_date: typing.Optional[datetime] = None,
):
    settings = read_general_settings()
    data_dir = (
        pathlib.Path(settings.get("data_dir")).joinpath("TRTH").joinpath("parsed_data")
    ).expanduser()
    # TODO: walk based on start/end dates
    for dirpath, _, filenames in os.walk(data_dir.as_posix()):
        security = pathlib.Path(dirpath).name
        for filename in filenames:
            if ".csv.gz" not in filename:
                continue
            date = filename.replace(".csv.gz", "")
            data = pd.read_csv(os.path.join(dirpath, filename), compression="gzip")
            yield date, security, data
            del data


def _load_data_for_security_date(
    security: str, date: datetime, data_dir: pathlib.Path
) -> pd.DataFrame:
    path = data_dir.joinpath(security, f"{date.strftime('%Y-%m-%d')}.csv.gz").as_posix()
    return pd.DataFrame(path)
