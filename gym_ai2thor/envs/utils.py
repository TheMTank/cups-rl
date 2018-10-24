"""
Auxiliary functions for building environments
"""
import os
import configparser


def read_config(config_path):
    """
    Returns the parsed information from the config file
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    config = configparser.ConfigParser()
    config_output = config.read(config_path)
    if not config_output:
        raise error.Error('No config file found at: {}. Exiting'.format(config_path))

    return config_output