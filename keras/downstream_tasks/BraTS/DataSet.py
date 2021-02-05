#!/usr/bin/env python
"""
File: DataSet
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)

This file provides loading of the BraTS datasets
for ease of use in TensorFlow models.
"""

import os

import pandas as pd
import numpy as np
import nibabel as nib

from tqdm import tqdm
from BraTS.Patient import *
from BraTS.structure import *
from BraTS.modalities import *
from BraTS.load_utils import *

survival_df_cache = {}  # Prevents loading CSVs more than once


class DataSubSet:

    def __init__(self, directory_map, survival_csv, data_set_type=None):
        self.directory_map = directory_map
        self._patient_ids = sorted(list(directory_map.keys()))
        self._survival_csv = survival_csv
        self._num_patients = len(self._patient_ids)
        self.type = data_set_type

        # Data caches
        self._mris = None
        self._segs = None
        self._patients = {}
        self._survival_df_cached = None
        self._patients_fully_loaded = False
        self._id_indexer = {patient_id: i for i, patient_id in enumerate(self._patient_ids)}

    def subset(self, patient_ids):
        """
        Split this data subset into a small subset by patient ID

        :param n: The number of elements in the smaller training set
        :return: A new data subset with only the specified number of items
        """
        dir_map = {id: self.directory_map[id] for id in patient_ids}
        return DataSubSet(dir_map, self._survival_csv)

    @property
    def ids(self):
        """
        List of all patient IDs in this dataset

        Will copy the ids... so modify them all you want
        :return: Copy of the patient IDs
        """
        return list(self._patient_ids)

    @property
    def mris(self):
        if self._mris is not None:
            return self._mris
        self._load_images()
        return self._mris

    @property
    def segs(self):
        if self._segs is None:
            self._load_images()
        return self._segs

    def _load_images(self):
        mris_shape = (self._num_patients,) + mri_shape
        segs_shape = (self._num_patients,) + image_shape

        self._mris = np.empty(shape=mris_shape)
        self._segs = np.empty(shape=segs_shape)

        if self._patients_fully_loaded:
            # All the patients were already loaded
            for i, patient in enumerate(tqdm(self._patients.values())):
                self._mris[i] = patient.mri_data
                self._segs[i] = patient.seg
        else:
            # Load it from scratch
            for i, patient_id in enumerate(self._patient_ids):
                patient_dir = self.directory_map[patient_id]
                load_patient_data_inplace(patient_dir, self._mris, self._segs, i)

    @property
    def patients(self):
        """
        Loads ALL of the patients from disk into patient objects

        :return: A dictionary containing ALL patients
        """
        for patient_id in self.ids:
            yield self.patient(patient_id)
        self._patients_fully_loaded = True

    def patient(self, patient_id):
        """
        Loads only a single patient from disk

        :param patient_id: The patient ID
        :return: A Patient object loaded from disk
        """
        if patient_id not in self._patient_ids:
            raise ValueError("Patient id \"%s\" not present." % patient_id)

        # Return cached value if present
        if patient_id in self._patients:
            return self._patients[patient_id]

        # Load patient data into memory
        patient = Patient(patient_id)
        patient_dir = self.directory_map[patient_id]

        df = self._survival_df
        if patient_id in df.id.values:
            patient.age = float(df.loc[df.id == patient_id].age)
            patient.survival = int(df.loc[df.id == patient_id].survival)

        if self._mris is not None and self._segs is not None:
            # Load from _mris and _segs if possible
            index = self._id_indexer[patient_id]
            patient.mri = self._mris[index]
            patient.seg = self._segs[index]
        else:
            # Load the mri and segmentation data from disk
            patient.mri, patient.seg = load_patient_data(patient_dir)

        self._patients[patient_id] = patient  # cache the value for later
        return patient

    def drop_cache(self):
        self._patients.clear()
        self._mris = None
        self._segs = None


    @property
    def _survival_df(self):
        if self._survival_csv in survival_df_cache:
            return survival_df_cache[self._survival_csv]
        df = load_survival(self._survival_csv)
        survival_df_cache[self._survival_csv] = df
        return df


class DataSet(object):

    def __init__(self, data_set_dir=None, brats_root=None, year=None):

        if data_set_dir is not None:
            # The data-set directory was specified explicitly
            assert isinstance(data_set_dir, str)
            self._data_set_dir = data_set_dir

        elif brats_root is not None and isinstance(year, int):
            # Find the directory by specifying the year
            assert isinstance(brats_root, str)
            year_dir = find_file_containing(brats_root, str(year % 100))
            self._data_set_dir = os.path.join(brats_root, year_dir)
            self._brats_root = brats_root
            self._year = year

        else:
            # BraTS data-set location was not improperly specified
            raise Exception("Specify BraTS location with \"data_set_dir\" or with \"brats_root\" and \"year\"")

        self._validation = None
        self._train = None
        self._hgg = None
        self._lgg = None

        self._dir_map_cache = None

        self._val_dir = None
        self._train_dir_cached = None
        self._hgg_dir = os.path.join(self._train_dir, "HGG")
        self._lgg_dir = os.path.join(self._train_dir, "LGG")

        self._train_survival_csv_cached = None
        self._validation_survival_csv_cached = None

        self._train_ids = None
        self._hgg_ids_cached = None
        self._lgg_ids_cached = None

        self._train_dir_map_cache = None
        self._validation_dir_map_cache = None
        self._hgg_dir_map_cache = None
        self._lgg_dir_map_cache = None

    def set(self, data_set_type):
        """
        Get a data subset by type

        :param data_set_type: The DataSubsetType to get
        :return: The data sub-set of interest
        """
        assert isinstance(data_set_type, DataSubsetType)
        if data_set_type == DataSubsetType.train:
            return self.train
        if data_set_type == DataSubsetType.hgg:
            return self.hgg
        if data_set_type == DataSubsetType.lgg:
            return self.lgg
        if data_set_type == DataSubsetType.validation:
            return self.validation

    @property
    def train(self):
        """
        Training data

        Loads the training data from disk, utilizing caching
        :return: A tf.data.Dataset object containing the training data
        """
        if self._train is None:
            try:
                self._train = DataSubSet(self._train_dir_map,
                                         self._train_survival_csv,
                                         data_set_type=DataSubsetType.train)
            except FileNotFoundError:
                return None
        return self._train

    @property
    def validation(self):
        """
        Validation data

        :return: Validation data
        """
        if self._validation is None:
            try:
                self._validation = DataSubSet(self._validation_dir_map,
                                              self._validation_survival_csv,
                                              data_set_type=DataSubsetType.validation)
            except FileNotFoundError:
                return None
        return self._validation

    @property
    def hgg(self):
        if self._hgg is None:
            try:
                self._hgg = DataSubSet(self._hgg_dir_map,
                                       self._train_survival_csv,
                                       data_set_type=DataSubsetType.hgg)
            except FileNotFoundError:
                return None
        return self._hgg

    @property
    def lgg(self):
        if self._lgg is None:
            try:
                self._lgg = DataSubSet(self._lgg_dir_map,
                                       self._train_survival_csv,
                                       data_set_type=DataSubsetType.lgg)
            except FileNotFoundError:
                return None
        return self._lgg

    def drop_cache(self):
        """
        Drops the cached values in the object
        :return: None
        """
        self._validation = None
        self._train = None
        self._hgg = None
        self._lgg = None

        self._dir_map_cache = None
        self._val_dir = None
        self._train_dir_cached = None
        self._train_survival_csv_cached = None
        self._validation_survival_csv_cached = None

        self._train_ids = None
        self._hgg_ids_cached = None
        self._lgg_ids_cached = None

        self._train_dir_map_cache = None
        self._validation_dir_map_cache = None
        self._hgg_dir_map_cache = None
        self._lgg_dir_map_cache = None

    @property
    def _train_survival_csv(self):
        if self._train_survival_csv_cached is None:
            self._train_survival_csv_cached = find_file_containing(self._train_dir, "survival")
            if self._train_survival_csv_cached is None:
                raise FileNotFoundError("Could not find survival CSV in %s" % self._train_dir)
        return self._train_survival_csv_cached

    @property
    def _validation_survival_csv(self):
        if self._validation_survival_csv_cached is None:
            self._validation_survival_csv_cached = find_file_containing(self._validation_dir, "survival")
            if self._validation_survival_csv_cached is None:
                raise FileNotFoundError("Could not find survival CSV in %s" % self._validation_dir)
        return self._validation_survival_csv_cached

    @property
    def _train_dir(self):
        if self._train_dir_cached is not None:
            return self._train_dir_cached
        self._train_dir_cached = find_file_containing(self._data_set_dir, "training")
        if self._train_dir_cached is None:
            raise FileNotFoundError("Could not find training directory in %s" % self._data_set_dir)
        return self._train_dir_cached

    @property
    def _validation_dir(self):
        if self._val_dir is not None:
            return self._val_dir
        self._val_dir = find_file_containing(self._data_set_dir, "validation")
        if self._val_dir is None:
            raise FileNotFoundError("Could not find validation directory in %s" % self._data_set_dir)
        return self._val_dir

    @property
    def _train_dir_map(self):
        if self._train_dir_map_cache is None:
            self._train_dir_map_cache = dict(self._hgg_dir_map)
            self._train_dir_map_cache.update(self._lgg_dir_map)
        return self._train_dir_map_cache

    @property
    def _validation_dir_map(self):
        if self._validation_dir_map_cache is None:
            self._validation_dir_map_cache = self._directory_map(self._validation_dir)
        return self._validation_dir_map_cache

    @property
    def _hgg_dir_map(self):
        if self._hgg_dir_map_cache is None:
            self._hgg_dir_map_cache = self._directory_map(self._hgg_dir)
        return self._hgg_dir_map_cache

    @property
    def _lgg_dir_map(self):
        if self._lgg_dir_map_cache is None:
            self._lgg_dir_map_cache = self._directory_map(self._lgg_dir)
        return self._lgg_dir_map_cache

    @property
    def _hgg_ids(self):
        if self._hgg_ids_cached is None:
            self._hgg_ids_cached = os.listdir(self._hgg_dir)
        return self._hgg_ids_cached

    @property
    def _lgg_ids(self):
        if self._lgg_ids_cached is None:
            self._lgg_ids_cached = os.listdir(self._lgg_dir)
        return self._lgg_ids_cached

    @classmethod
    def _directory_map(cls, dir):
        return {file: os.path.join(dir, file)
                 for file in os.listdir(dir)
                 if os.path.isdir(os.path.join(dir, file))}
