#!/usr/bin/env python
"""
File: utils
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import pandas as pd

def load_survival(survival_csv):
    """
    Loads a survival CSV file
    :param survival_csv: The path to the CSV file to load
    :return: Pandas DataFrame with the survival information
    """
    try:
        survival = pd.read_csv(survival_csv)
    except:
        raise Exception("Error reading survival CSV file: %s" % survival_csv)
    return rename_columns(survival)


def rename_columns(df):
    """
    Rename the columns of a survival data CSV so that they are consistent
    across different data-sets
    :param df: The raw Pandas DataFrame read from the survival CSV file
    :return: The same DataFrame but with the columns modified
    """
    if df.shape[1] == 3:
        df.columns = ['id', 'age', 'survival']
    elif df.shape[1] == 4:
        df.columns = ['id', 'age', 'survival', 'resection']
    else:
        raise Exception("Unknown columns in survival: %s" % df.columns)
    return df


def find_file_containing(directory, keyword, case_sensitive=False):
    """
    Finds a file in a directory containing a keyword in it's name

    :param directory: The directory to search in
    :param keyword: The keyword to search in the name for
    :param case_sensitive: Search with case sensitivity
    :return: The joined path to the file containing the keyword in
    it's name, if found, else None.
    """
    assert isinstance(directory, str)
    assert isinstance(keyword, str)

    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)

    # Iterate through files
    for file in os.listdir(directory):
        if keyword in (file if case_sensitive else file.lower()):
            return os.path.join(directory, file)
    return None


def find_file_named(root, name):
    """
    Find a file named something

    :param root: Root directory to search recursively through
    :param name: The name of the file to search for
    :return: Full path to the (first!) file with the specified name found,
    or None if no file was found of that name.
    """
    assert isinstance(root, str)
    assert isinstance(name, str)

    # Search the directory recursively
    for path, dirs, files in os.walk(root):
        for file in files:
            if file == name:
                return os.path.join(path, file)
    return None


def listdir(directory):
    """
    Gets the full paths to the contents of a directory

    :param directory: A path to some directory
    :return: An iterator yielding full paths to all files in the specified directory
    """
    assert isinstance(directory, str)
    m = map(lambda d: os.path.join(directory, d), os.listdir(directory))
    contents = [f for f in m if not f.startswith('.')]
    return contents
