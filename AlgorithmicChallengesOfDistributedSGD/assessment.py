from __future__ import print_function
import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(966)
rn.seed(966)
import argparse
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import backend as K
from keras_contrib.applications.wide_resnet import WideResidualNetwork
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from time import time
import os
import horovod.keras as hvd

hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
config.intra_op_parallelism_threads=1
config.inter_op_parallelism_threads=1
K.set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Keras Fashion MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.8,
                    help='SGD momentum')

args = parser.parse_args()

verbose = 0
if hvd.rank() == 0:
    verbose = 1
else:
    verbose = 0
num_classes = 10
data_augmentation = True

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions
img_rows, img_cols = 32, 32
num_classes = 10

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = WideResidualNetwork(depth=16, width=10, weights=None, input_shape=input_shape,
                                    classes=num_classes, dropout_rate=0.01)

opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Callbacks. Do NOT edit these, the assessment
# code depends on these appearing exactly as
# below, and you will fail the assessment if
# these are changed.

class PrintTotalTime(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = round(time() - self.start_time, 2)
        print("Elapsed training time through epoch {}: {}".format(epoch+1, elapsed_time))

    def on_train_end(self, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Total training time: {}".format(total_time)) 

class StopAtAccuracy(keras.callbacks.Callback):
    def __init__(self, train_target=0.75, val_target=0.25, patience=2, verbose=0):
        self.train_target = train_target
        self.val_target = val_target
        self.patience = patience
        self.verbose = verbose
        self.stopped_epoch = 0
        self.met_train_target = 0
        self.met_val_target = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc') > self.train_target:
            self.met_train_target += 1
        else:
            self.met_train_target = 0
            
        if logs.get('val_acc') > self.val_target:
            self.met_val_target += 1
        else:
            self.met_val_target = 0

        if self.met_train_target >= self.patience and self.met_val_target >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and verbose == 1:
            print('Early stopping after epoch {}. Training accuracy target ({}) and validation accuracy target ({}) met.'.format(self.stopped_epoch + 1, self.train_target, self.val_target))

def lr_schedule(epoch):
    
    if epoch < 15:
        return args.base_lr
    if epoch < 25:
        return 1e-1 * args.base_lr
    if epoch < 35:
        return 1e-2 * args.base_lr
    return 1e-3 * args.base_lr

callbacks = [
    StopAtAccuracy(verbose=verbose),
    keras.callbacks.LearningRateScheduler(lr_schedule)
]

if verbose:
    callbacks.append(PrintTotalTime())
    
callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
callbacks.append(hvd.callbacks.MetricAverageCallback())

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
train_iter = datagen.flow(x_train, y_train, batch_size=args.batch_size)

datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(train_iter,
                    callbacks=callbacks,
                    epochs=args.epochs,
                    verbose=verbose,
                    # avoid shuffling for reproducible training
                    shuffle=False,
                    steps_per_epoch=len(train_iter) // hvd.size(),
                    validation_data=(x_test, y_test),
                    workers=4)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=verbose)
if verbose:
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])