#!/usr/bin/env python
"""
File: Patient
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""


import numpy as np
import nibabel as nib

from BraTS.modalities import *
from BraTS.load_utils import *


def load_patient_data(patient_data_dir):
    """
    Load a single patient's image data

    :param patient_data_dir: Directory containing image data
    :return: Tuple containing a tf.Tensor containing MRI data
    """
    mri_data = np.empty(shape=mri_shape)
    seg_data = None
    for img_file in listdir(patient_data_dir):
        img = nib.load(img_file).get_data()
        img_type = get_modality(img_file)

        if img_type == Modality.seg:
            seg_data = img
        else:
            channel_index = modality_indices[img_type]
            mri_data[channel_index] = img
    return mri_data, seg_data  # Load segmentation data


def load_patient_data_inplace(patient_data_dir, mri_array, seg_array, index):
    """
    Loads patient data into an existing mri array

    :param patient_data_dir: Directory containing patient images
    :param mri_array: Array to load the patient MRI into
    :param seg_array: Array to load the patient segmentation into
    :param index: Index of mri_array and seg_array to load into
    :return: None
    """
    for img_file in listdir(patient_data_dir):
        img = nib.load(img_file).get_data()
        img_type = get_modality(img_file)
        if img_type == Modality.seg:
            seg_array[index] = img
            continue
        else:
            channel_index = modality_indices[img_type]
            mri_array[index, channel_index] = img


class Patient:
    def __init__(self, id, age=None, survival=None, mri=None, seg=None):
        self.id = id
        self.age = age
        self.survival = survival
        self.mri = mri
        self.seg = seg

    @property
    def flair(self):
        if not isinstance(self.mri, np.ndarray):
            raise Exception("patient %s MRI is not a numpy array." % self.id)
        return self.mri[0]

    @property
    def t1(self):
        if not isinstance(self.mri, np.ndarray):
            raise Exception("patient %s MRI is not a numpy array." % self.id)
        return self.mri[1]

    @property
    def t1ce(self):
        if not isinstance(self.mri, np.ndarray):
            raise Exception("patient %s MRI is not a numpy array." % self.id)
        return self.mri[2]

    @property
    def t2(self):
        if not isinstance(self.mri, np.ndarray):
            raise Exception("patient %s MRI is not a numpy array." % self.id)
        return self.mri[3]
