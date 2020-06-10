import os
import pathlib
from configparser import ConfigParser, ExtendedInterpolation

config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_path)

conf = config['Paths']
data_dir = pathlib.Path(conf['data_dir']).expanduser()
result_dir = pathlib.Path(conf['result_dir']).expanduser()

os.makedirs(data_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
