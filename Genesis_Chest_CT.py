#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

"""
CUDA_VISIBLE_DEVICES=1 python -W ignore Genesis_Chest_CT.py \
--note genesis_chest_ct \
--arch Vnet \
--input_rows 64 \
--input_cols 64 \
--input_deps 32 \
--nb_class 1 \
--verbose 1 \
--batch_size 16 \
--scale 32 \
--data generated_cubes
"""

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import os
import keras
print("Keras = {}".format(keras.__version__))
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


import copy
import scipy
import sys
import math
import random
import shutil
import string
import numpy as np

from tqdm import tqdm
from scipy.misc import comb
from sklearn import metrics
from unet3d import *
from keras.callbacks import LambdaCallback, TensorBoard
from skimage.transform import resize
from optparse import OptionParser
from keras.utils import plot_model

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--arch", dest="arch", help="Vnet", default=None, type="string")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=64, type="int")
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=64, type="int")
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int")
parser.add_option("--nb_class", dest="nb_class", help="number of class", default=1, type="int")
parser.add_option("--verbose", dest="verbose", help="verbose", default=0, type="int")
parser.add_option("--weights", dest="weights", help="pre-trained weights", default=None, type="string")
parser.add_option("--note", dest="note", help="notes of experiment setup", default="", type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=8, type="int")
parser.add_option("--scale", dest="scale", help="the scale of pre-trained data", default=32, type="int")
parser.add_option("--optimizer", dest="optimizer", help="SGD | Adam", default="Adam", type="string")
parser.add_option("--data", dest="data", help="the address of data cube", default=None, type="string")
parser.add_option("--workers", dest="workers", help="number of CPU cores", default=8, type="int")
parser.add_option("--save_samples", dest="save_samples", help="None | png | jpg", default="png", type="string")

parser.add_option("--nonlinear_rate", dest="nonlinear_rate", help="chance to perform nonlinear", default=0.9, type="float")
parser.add_option("--paint_rate", dest="paint_rate", help="chance to perform painting", default=0.9, type="float")
parser.add_option("--outpaint_rate", dest="outpaint_rate", help="chance to perform out-painting", default=0.8, type="float")
parser.add_option("--flip_rate", dest="flip_rate", help="chance to perform flipping", default=0.9, type="float")
parser.add_option("--local_rate", dest="local_rate", help="chance to perform local shuffle pixel", default=0.1, type="float")

(options, args) = parser.parse_args()

assert options.arch in ['Vnet']
assert options.data is not None
assert os.path.exists(options.data) == True

seed = 1
random.seed(seed)
model_path = "pretrained_weights"
if not os.path.exists(model_path):
    os.makedirs(model_path)
logs_path = os.path.join(model_path, "Logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
sample_path = "pair_samples"
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
    
class setup_config():
    nb_epoch = 10000
    patience = 50
    lr = 1e-0
    train_fold=[0,1,2,3,4]
    valid_fold=[5,6]
    test_fold=[7,8,9]
    hu_max = 1000.0
    hu_min = -1000.0
    def __init__(self, model="Unet",
                 note="",
                 data_augmentation=True,
                 input_rows=64, 
                 input_cols=64,
                 input_deps=32,
                 batch_size=64,
                 nb_class=1,
                 nonlinear_rate=0.95,
                 paint_rate=0.6,
                 outpaint_rate=0.8,
                 flip_rate=0.0,
                 local_rate=0.9,
                 verbose=1,
                 workers=2,
                 optimizer=None,
                 DATA_DIR=None,
                 save_samples=None,
                ):
        self.model = model
        self.exp_name = model + "-" + note
        self.data_augmentation = data_augmentation
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.nonlinear_rate = nonlinear_rate
        self.paint_rate = paint_rate
        self.outpaint_rate = outpaint_rate
        self.inpaint_rate = 1.0 - self.outpaint_rate
        self.flip_rate = flip_rate
        self.local_rate = local_rate
        self.nb_class = nb_class
        self.optimizer = optimizer
        self.workers = workers
        self.DATA_DIR = DATA_DIR
        self.save_samples = save_samples
        self.max_queue_size = self.workers * 4
        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

config = setup_config(model=options.arch,
                      note=options.note,
                      input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      batch_size=options.batch_size,
                      nb_class=options.nb_class,
                      verbose=options.verbose,
                      nonlinear_rate=options.nonlinear_rate,
                      paint_rate=options.paint_rate,
                      outpaint_rate=options.outpaint_rate,
                      flip_rate=options.flip_rate,
                      local_rate=options.local_rate,
                      optimizer=options.optimizer,
                      DATA_DIR=options.data,
                      workers=options.workers,
                      save_samples=options.save_samples,
                     )
config.display()
shutil.rmtree(os.path.join(sample_path, config.exp_name), ignore_errors=True)
if not os.path.exists(os.path.join(sample_path, config.exp_name)):
    os.makedirs(os.path.join(sample_path, config.exp_name))

# In[2]:

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
    return x

def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[:, 
      noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y, 
      noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y, 
                                                       noise_z:noise_z+block_noise_size_z]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                           noise_y:noise_y+block_noise_size_y, 
                                                           noise_z:noise_z+block_noise_size_z]
    return x
                


def generate_pair(img, batch_size, status="test"):
    img_rows, img_cols, img_deps = img.shape[2], img.shape[3], img.shape[4]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):
            
            # Autoencoder
            x[n] = copy.deepcopy(y[n])
            
            # Flip
            x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

            # Local Shuffle Pixel
            x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
            
            # Apply non-Linear transformation with an assigned probability
            x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
            
            # Inpainting & Outpainting
            if random.random() < config.paint_rate:
                if random.random() < config.inpaint_rate:
                    # Inpainting
                    x[n] = image_in_painting(x[n])
                else:
                    # Outpainting
                    x[n] = image_out_painting(x[n])

        # Save sample images module
        if config.save_samples is not None and status == "train" and random.random() < 0.01:
            n_sample = random.choice( [i for i in range(config.batch_size)] )
            sample_1 = np.concatenate((x[n_sample,0,:,:,2*img_deps//6], y[n_sample,0,:,:,2*img_deps//6]), axis=1)
            sample_2 = np.concatenate((x[n_sample,0,:,:,3*img_deps//6], y[n_sample,0,:,:,3*img_deps//6]), axis=1)
            sample_3 = np.concatenate((x[n_sample,0,:,:,4*img_deps//6], y[n_sample,0,:,:,4*img_deps//6]), axis=1)
            sample_4 = np.concatenate((x[n_sample,0,:,:,5*img_deps//6], y[n_sample,0,:,:,5*img_deps//6]), axis=1)
            final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
            scipy.misc.imsave(os.path.join(sample_path, config.exp_name, 
                                           ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples), 
                              final_sample)

        yield (x, y)


# learning rate schedule
# source: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay(epoch):
    
    initial_lrate = config.lr
    drop = 0.5
    epochs_drop = int(config.patience * 0.2)
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    return lrate


x_train = []
for i,fold in enumerate(tqdm(config.train_fold)):
    s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x32_"+str(fold)+".npy"))
    x_train.extend(s)
x_train = np.expand_dims(np.array(x_train), axis=1)
print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))

x_valid = []
for i,fold in enumerate(tqdm(config.valid_fold)):
    s = np.load(os.path.join(config.DATA_DIR, "bat_"+str(options.scale)+"_s_64x64x32_"+str(fold)+".npy"))
    x_valid.extend(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))


if config.model == "Vnet":
    model = unet_model_3d((1, config.input_rows, config.input_cols, config.input_deps), batch_normalization=True)
if options.weights is not None:
    print("Load the pre-trained weights from {}".format(options.weights))
    model.load_weights(options.weights)

# plot_model(model, to_file=os.path.join(model_path, config.exp_name+".png"))
if os.path.exists(os.path.join(model_path, config.exp_name+".txt")):
    os.remove(os.path.join(model_path, config.exp_name+".txt"))
with open(os.path.join(model_path, config.exp_name+".txt"),'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

shutil.rmtree(os.path.join(logs_path, config.exp_name), ignore_errors=True)
if not os.path.exists(os.path.join(logs_path, config.exp_name)):
    os.makedirs(os.path.join(logs_path, config.exp_name))
tbCallBack = TensorBoard(log_dir=os.path.join(logs_path, config.exp_name),
                         histogram_freq=0,
                         write_graph=True, 
                         write_images=True,
                        )
tbCallBack.set_model(model)    

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                               patience=config.patience, 
                                               verbose=0,
                                               mode='min',
                                              )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(model_path, config.exp_name+".h5"),
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only=True, 
                                              mode='min',
                                             )
if config.optimizer == "SGD" or config.optimizer == "sgd":
    model.compile(optimizer=keras.optimizers.SGD(lr=config.lr, momentum=0.9, decay=0.0, nesterov=False), 
                  loss="MSE", 
                  metrics=["MAE", "MSE"])
    lrate = keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
    callbacks = [check_point, early_stopping, tbCallBack, lrate]
elif config.optimizer == "Adam" or config.optimizer == "adam":
    model.compile(optimizer="Adam", 
                  loss="MSE", 
                  metrics=["MAE", "MSE"])
    callbacks = [check_point, early_stopping, tbCallBack]
else:
    raise


while config.batch_size > 1:
    # To find a largest batch size that can be fit into GPU
    try:
        model.fit_generator(generate_pair(x_train, config.batch_size, status="train"),
                            validation_data=generate_pair(x_valid, config.batch_size, status="test"), 
                            validation_steps=int(2.0*x_valid.shape[0]//config.batch_size),
                            steps_per_epoch=x_train.shape[0]//config.batch_size, 
                            epochs=config.nb_epoch,
                            max_queue_size=config.max_queue_size, 
                            workers=config.workers, 
                            use_multiprocessing=True, 
                            shuffle=True,
                            verbose=config.verbose, 
                            callbacks=callbacks,
                           )
        break
    except tf.errors.ResourceExhaustedError as e:
        config.batch_size = int(config.batch_size - 2)
        print("\n> Batch size = {}".format(config.batch_size))
