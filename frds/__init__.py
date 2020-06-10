import os
import sys
import pathlib
from configparser import ConfigParser, ExtendedInterpolation

if sys.version_info.major < 3 and sys.version_info.minor < 8:
    print('Python3.8 and higher is required.')
    sys.exit(1)

# default config.ini comes with the frds module
default_config = os.path.join(os.path.dirname(__file__), 'config.ini')

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(default_config)
base_dir = pathlib.Path(config['Paths']['base_dir']).expanduser()

# custom config.ini in user's directory
custom_config = os.path.join(base_dir, 'config.ini')
config.read(custom_config)

data_dir = pathlib.Path(config['Paths']['data_dir']).expanduser()
result_dir = pathlib.Path(config['Paths']['result_dir']).expanduser()

os.makedirs(data_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

conf = config['Login']
wrds_username = conf.get('wrds_username', fallback=None)
wrds_password = conf.get('wrds_password', fallback=None)

if not all([wrds_username, wrds_password]):
    print(f'Please enter your WRDS username and password in {custom_config}.')
