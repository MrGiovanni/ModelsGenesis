#!/usr/bin/env python
"""
File: structure
Date: 5/8/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
from enum import Enum

from BraTS.load_utils import find_file_containing


class DataSubsetType(Enum):
    hgg = 0
    lgg = 1
    train = 2
    validation = 3


def get_brats_subset_directory(brats_dataset_dir, data_set_type):
    if data_set_type == DataSubsetType.train:
        # Training data
        try:
            found_train = find_file_containing(brats_dataset_dir, "train", case_sensitive=False)
        except FileNotFoundError:
            found_train = None
        if found_train is not None:
            return found_train
        return os.path.join(brats_dataset_dir, "training")

    if data_set_type == DataSubsetType.hgg:
        train_dir = get_brats_subset_directory(brats_dataset_dir, DataSubsetType.train)
        return os.path.join(train_dir, "HGG")

    if data_set_type == DataSubsetType.lgg:
        train_dir = get_brats_subset_directory(brats_dataset_dir, DataSubsetType.train)
        return os.path.join(train_dir, "LGG")

    if data_set_type == DataSubsetType.validation:
        # Validation
        try:
            found_validation = find_file_containing(brats_dataset_dir, "validation", case_sensitive=False)
        except FileNotFoundError:
            found_validation = None

        if found_validation is not None:
            return found_validation
        return os.path.join(brats_dataset_dir, "validation")

