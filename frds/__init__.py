import os
import sys
import pathlib
from configparser import ConfigParser, ExtendedInterpolation

if sys.version_info.major < 3 and sys.version_info.minor < 8:
    print("Python3.8 and higher is required.")
    sys.exit(1)

config = ConfigParser(interpolation=ExtendedInterpolation())

# custom config.ini in user's directory
custom_config = os.path.join(pathlib.Path("~/frds").expanduser(), "config.ini")
config.read(custom_config)

data_dir = pathlib.Path(config["Paths"]["data_dir"]).expanduser()
result_dir = pathlib.Path(config["Paths"]["result_dir"]).expanduser()

os.makedirs(data_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

credentials = config._sections["Login"]

