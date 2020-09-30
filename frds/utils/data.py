import typing
import pathlib
from datetime import datetime
import pandas as pd
from frds.utils.settings import read_general_settings


# Move these two below to a util function
def get_market_microstructure_data(
    securities: typing.List[str], start_date: datetime, end_date: datetime
) -> typing.Iterator[typing.Tuple[datetime, str, pd.DataFrame]]:
    """Yield (date, security, data)"""

    settings = read_general_settings()
    data_dir = (
        pathlib.Path(settings.get("data_dir")).joinpath("TRTH").joinpath("parsed_data")
    )

    for date in pd.date_range(start_date, end_date):
        for security in securities:
            data = _load_data_for_security_date(security, date, data_dir)
            yield date, security, data
            del data


def _load_data_for_security_date(
    security: str, date: datetime, data_dir: pathlib.Path
) -> pd.DataFrame:
    path = data_dir.joinpath(security, f"{date.strftime('%Y-%m-%d')}.csv.gz").as_posix()
    return pd.DataFrame(path)
