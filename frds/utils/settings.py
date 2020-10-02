from typing import Dict
import functools
import os
import pathlib
import configparser

from frds.settings import CONFIG_FILE_PATH, CONFIG_FILE_NAME


def _get_config_file_path(path=CONFIG_FILE_PATH) -> pathlib.Path:
    path = pathlib.Path(path).expanduser()
    if not path.is_dir:
        os.makedirs(path.as_posix(), exist_ok=True)
    return path.joinpath(CONFIG_FILE_NAME)


@functools.lru_cache()
def _read_config_file() -> Dict:
    path = _get_config_file_path()
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read(path)
    return {section: config[section] for section in config.sections()}


def save_config_to_file(config_to_save: Dict) -> None:
    current_config = _read_config_file()
    config = configparser.ConfigParser()
    current_config.update(config_to_save)
    config.read_dict(current_config)
    path = _get_config_file_path()
    with open(path.as_posix(), "w") as f:
        config.write(f)


def read_data_source_credentials() -> Dict:
    config = _read_config_file()
    return config.get("Credentials", {})


def save_data_source_credentials(credentials: Dict) -> None:
    config = _read_config_file()
    config["Credentials"] = credentials
    save_config_to_file(config)


def read_general_settings() -> Dict:
    config = _read_config_file()
    return config.get("General", {})


def save_general_settings(general_settings: Dict) -> None:
    config = _read_config_file()
    config["General"] = general_settings
    save_config_to_file(config)
