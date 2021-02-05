#!/usr/bin/env python
"""
File: modalities
Date: 5/4/18 
Author: Jon Deaton (jdeaton@stanford.edu)

Generic information about the different image types
available in the BraTS data sets.
"""

from BraTS.load_utils import *

from enum import Enum

# Each of the "images" is a 3D volume of this shape
image_shape = (240, 240, 155)

# Here, "MRI" refers to all four modalities taken during the scan
mri_shape = (4,) + image_shape

# The segmentation map is just an image but with pixels indicating which
seg_shape = image_shape


class Modality(Enum):
    """
    Enumeration representing the different types of modalities/images available
    in the BraTS data set (T1, T2, T1CE, Flair, and Segmentation).
    Segmentation isn't technically an image but it's convenient
    to have it as a member of this enum because its stored in the data set
    as another file just as though it were an image.
    """
    t1 = 1
    t2 = 2
    t1ce = 3
    flair = 4
    seg = 5


# Iterable, ordered collection of image types
modalities = [Modality.t1, Modality.t2, Modality.t1ce,
              Modality.flair, Modality.seg]

# Map from image type to name
modality_names = {Modality.t1: "t1",
                  Modality.t2: "t2",
                  Modality.t1ce: "t1ce",
                  Modality.flair: "flair",
                  Modality.seg: "seg"}

"""
The MRI is a 4D Numpy array. The first dimension
indices through each of the four image types. This map
codifies which index in that first dimension is dedicated
to which image type (for consistency)
"""
modality_indices = {Modality.t1: 0,
                    Modality.t2: 1,
                    Modality.t1ce: 2,
                    Modality.flair: 3}


def get_modality(image_file):
    """
    Determines the image type of a file from it's name

    :param image_file: File name or file path
    :return: The type of image if it could be determined, None otherwise
    """
    for img_type, name in modality_names.items():
        if "%s." % name in os.path.basename(image_file):
            return img_type
    return None


def get_modality_map(patient_dir):
    """
    Creates a map for finding a given type of image within
    a patien's directory of images

    :param patient_dir: Directory of patient images
    :return: Dictionary mapping image type to file path for the image
    of that type
    """
    files = listdir(patient_dir)
    d = {}
    for file in files:
        d[get_modality(file)] = file
    return d

def get_modality_file(patient_dir, modality):
    return get_modality_map(patient_dir)[modality]
