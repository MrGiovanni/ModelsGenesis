import os
import random
import copy
import keras
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from skimage.transform import resize

def load_image(config, status=None):
    
    x = np.squeeze(np.load(os.path.join(config.data, 'x_'+status+'_64x64x32.npy')))
    y = np.squeeze(np.load(os.path.join(config.data, 'm_'+status+'_64x64x32.npy')))
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)

    return x, y

if __name__ == "__main__":
    from config import *

    class set_args():
        apps = 'ncs'
        task = 'segmentation'
        suffix = 'random'
    args = set_args()

    conf = ncs_config(args)
    x_valid, y_valid = load_image(conf, 'valid')
    print(x_valid.shape, np.min(x_valid), np.max(x_valid))
    print(y_valid.shape, np.min(y_valid), np.max(y_valid))