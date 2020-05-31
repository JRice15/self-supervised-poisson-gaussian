""" Train network on the Fluorescence Microscopy Dataset (FMD) """ 

import argparse
import glob
import os
from os import listdir
from os.path import join

import keras
import numpy as np
import skimage
import tensorflow as tf
from imageio import imread
from keras import backend as K
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from tqdm import trange

from callback import LogProgress
from nets import *

np.random.seed(1234)
tf.set_random_seed(1234)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to dataset root')
parser.add_argument('--dataset',required=True,help='dataset name e.g. Confocal_MICE')
parser.add_argument('--mode',default='uncalib',help='noise model: uncalib, gaussian, poisson, or poissongaussian')
parser.add_argument('--reg',type=float,default=10,help='regularization weight on prior std. dev.')
parser.add_argument('--crop',type=int,default=128,help='crop size')
parser.add_argument('--batch',type=int,default=4,help='batch size')
parser.add_argument('--epoch',type=int,default=300,help='num epochs')
parser.add_argument('--steps',type=int,default=50,help='steps per epoch')
parser.add_argument('--lr',type=float,default=0.0003,help='learning rate')
parser.add_argument('--components',type=int,default=1,help='number of mixture components')
parser.add_argument('--patience',type=int,default=10,help='ReduceLROnPlateau patience')
parser.add_argument('--tag',type=str,default="",help='id tag to add to weights path')
parser.add_argument('--width',type=int,default=1,help='blindspot width')
parser.add_argument('--height',type=int,default=1,help='blindspot height')

args = parser.parse_args()

if args.components != 1 and args.mode != "uncalib":
    raise ValueError("Components != 1 must be used with mode uncalib")    

if args.width % 2 != 1 or args.height % 2 != 1:
    raise ValueError("Blindspot width and height must be odd integers")

""" Load dataset """

def load_images(noise):
    basepath = args.path + '/' + args.dataset + '/' + noise
    images = []
    for i in range(1,21):
        if i==19: continue
        for path in sorted(glob.glob(basepath + '/%d/*.png'%i)):
            images.append(imread(path))
    return np.stack(images,axis=0)[:,:,:,None]/255.

train_images = load_images('raw')
np.random.shuffle(train_images)

X = train_images[:-5]
X_val = train_images[-5:]
print('%d training images'%len(X))
print('%d validation images'%len(X_val))

""" Augment by rotating and flipping """
""" Adapted from https://github.com/juglab/n2v/blob/master/n2v/internals/N2V_DataGenerator.py """

def augment_images(images):
  augmented = np.concatenate((images,
                              np.rot90(images, k=1, axes=(1, 2)),
                              np.rot90(images, k=2, axes=(1, 2)),
                              np.rot90(images, k=3, axes=(1, 2))))
  augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
  return augmented

X = augment_images(X)
X_val = augment_images(X_val)
print('%d training images after augmenting'%len(X))

""" Training """
""" Train on random crops of the training image."""

def random_crop_generator(data, crop_size, batch_size):
    while True:
        inds = np.random.randint(data.shape[0],size=batch_size)
        y = np.random.randint(data.shape[1]-crop_size,size=batch_size)
        x = np.random.randint(data.shape[2]-crop_size,size=batch_size)
        batch = np.zeros((batch_size,crop_size,crop_size,1),dtype=data.dtype)
        for i,ind in enumerate(inds):
            batch[i] = data[ind,y[i]:y[i]+crop_size,x[i]:x[i]+crop_size]
        yield batch, None


model = gaussian_blindspot_network(
    (args.crop, args.crop, 1),
    args.mode,
    args.reg,
    components=args.components,
    width=args.width,
    height=args.height
)

model.compile(optimizer=Adam(args.lr))

os.makedirs('weights',exist_ok=True)

experiment_name = '%s.%s'%(args.dataset,args.mode)
if args.tag != "":
    experiment_name += '.%s'%(args.tag)
if args.width != 1 or args.height != 1:
    experiment_name += '.%dx%d'%(args.width, args.height)
if args.mode == 'uncalib' or args.mode == 'mse':
    if args.components != 1:
        experiment_name += '.%dcomponents'%(args.components)
else:
    experiment_name += '.%0.3f'%(args.reg)

weights_path = "weights/weights." + experiment_name + ".latest.hdf5"

callbacks = []
callbacks.append(ModelCheckpoint(filepath=weights_path, monitor='val_loss',save_best_only=1,verbose=1))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=args.patience, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))
callbacks.append(LogProgress(experiment_name))

gen = random_crop_generator(X,args.crop,args.batch)
val_crops = []
for y in range(0,X_val.shape[1],args.crop):
    if y+args.crop > X_val.shape[1]: continue
    for x in range(0,X_val.shape[2],args.crop):
        if x+args.crop > X_val.shape[2]: continue
        val_crops.append(X_val[:,y:y+args.crop,x:x+args.crop])
val_data = np.concatenate(val_crops,axis=0)

history = model.fit_generator(gen,
                              steps_per_epoch=args.steps,
                              validation_data=(val_data,None),
                              epochs=args.epoch, 
                              verbose=1,
                              callbacks=callbacks)
