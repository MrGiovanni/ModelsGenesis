#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import keras
print("keras = {}".format(keras.__version__))
import tensorflow as tf
print("tensorflow-gpu = {}".format(tf.__version__))
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

import shutil
import numpy as np
from tqdm import tqdm
from config import models_genesis_config
from utils import *
from unet3d import *
from keras.callbacks import LambdaCallback,TensorBoard,ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

conf = models_genesis_config()
conf.display()


# # Load subvolumes for self-supervised learning

# In[2]:


x_train = []
for i,fold in enumerate(tqdm(conf.train_fold)):
    file_name = "bat_"+str(conf.scale)+"_s_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    s = np.load(os.path.join(conf.data, file_name))
    x_train.extend(s)
x_train = np.expand_dims(np.array(x_train), axis=1)

x_valid = []
for i,fold in enumerate(tqdm(conf.valid_fold)):
    file_name = "bat_"+str(conf.scale)+"_s_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    s = np.load(os.path.join(conf.data, file_name))
    x_valid.extend(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)

print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))


# # Setup the model

# In[3]:


if conf.model == "Vnet":
    model = unet_model_3d((1, conf.input_rows, conf.input_cols, conf.input_deps), batch_normalization=True)
if conf.weights is not None:
    print("Load the pre-trained weights from {}".format(conf.weights))
    model.load_weights(conf.weights)
model.compile(optimizer=keras.optimizers.SGD(lr=conf.lr, momentum=0.9, decay=0.0, nesterov=False), 
              loss="MSE", 
              metrics=["MAE", "MSE"])

if os.path.exists(os.path.join(conf.model_path, conf.exp_name+".txt")):
    os.remove(os.path.join(conf.model_path, conf.exp_name+".txt"))
with open(os.path.join(conf.model_path, conf.exp_name+".txt"),'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

shutil.rmtree(os.path.join(conf.logs_path, conf.exp_name), ignore_errors=True)
if not os.path.exists(os.path.join(conf.logs_path, conf.exp_name)):
    os.makedirs(os.path.join(conf.logs_path, conf.exp_name))
tbCallBack = TensorBoard(log_dir=os.path.join(conf.logs_path, conf.exp_name),
                         histogram_freq=0,
                         write_graph=True, 
                         write_images=True,
                        )
tbCallBack.set_model(model)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                               patience=conf.patience, 
                                               verbose=0,
                                               mode='min',
                                              )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(conf.model_path, conf.exp_name+".h5"),
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only=True, 
                                              mode='min',
                                             )
lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                    min_delta=0.0001, min_lr=1e-6, verbose=1)

callbacks = [check_point, early_stopping, tbCallBack, lrate_scheduler]


# # Train Models Genesis

# In[ ]:


while conf.batch_size > 1:
    # To find a largest batch size that can be fit into GPU
    try:
        model.fit_generator(generate_pair(x_train, conf.batch_size, config=conf, status="train"),
                            validation_data=generate_pair(x_valid, conf.batch_size, config=conf, status="test"), 
                            validation_steps=x_valid.shape[0]//conf.batch_size,
                            steps_per_epoch=x_train.shape[0]//conf.batch_size, 
                            epochs=conf.nb_epoch,
                            max_queue_size=conf.max_queue_size, 
                            workers=conf.workers, 
                            use_multiprocessing=True, 
                            shuffle=True,
                            verbose=conf.verbose, 
                            callbacks=callbacks,
                           )
        break
    except tf.errors.ResourceExhaustedError as e:
        conf.batch_size = int(conf.batch_size - 2)
        print("\n> Batch size = {}".format(conf.batch_size))


# In[ ]:




