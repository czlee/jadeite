"""Configuration.

Reads the config.json file, and sets relevant attributes of itself accordingly.
Configuration variables should be accessed as, for example:

    from config import DATA_DIRECTORY, RESULTS_DIRECTORY

This uses some Python magic, but hopefully not too obscure. Its purpose is to
try to allow imports of the style above, without having to catch `ImportError`
everywhere the `config` module is used if `config.py` didn't exist.
"""

import pathlib

import yaml

# There aren't currently any optional configuration options, but this is called
# `_required_config_names` in case some get added later.
_required_config_names = [
    "DATA_DIRECTORY",
    "RESULTS_DIRECTORY",
]

_config_file = pathlib.Path("config.yaml")

if not _config_file.exists():
    print("Couldn't find config.yaml.")
    print("Copy config.yaml.example to config.yaml change the values in it to appropriate")
    print("values for your machine.")
    exit(1)

with open(_config_file) as f:
    _config_file_dict = yaml.safe_load(f)

if _config_file_dict is None:
    print("Nothing found in config.yaml.")
    print("Copy config.yaml.example to config.yaml change the values in it to appropriate")
    print("values for your machine.")
    exit(1)

# All options are in a top-level "config"
try:
    _config_options = _config_file_dict["config"]
except KeyError:
    print("Malformed config.yaml file.")
    exit(1)

_missing_names = set(_required_config_names) - set(_config_options.keys())

if _missing_names:
    print("The following required settings are missing from config.json:")
    for _name in _missing_names:
        print(" - " + _name)
    exit(1)

for _name in _required_config_names:
    globals()[_name] = _config_options[_name]

__all__ = _required_config_names
