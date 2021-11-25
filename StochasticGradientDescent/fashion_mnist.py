import argparse
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, \
                                    Add, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D
import numpy as np
import os
from time import time

# Parse input arguments

parser = argparse.ArgumentParser(description='Fashion MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--wd', type=float, default=0.000005,
                    help='weight decay')
# TODO Step 2: Add target and patience arguments to the argument parser

args = parser.parse_args()

# Define a function for a simple learning rate decay over time

def lr_schedule(epoch):
    
    if epoch < 15:
        return args.base_lr
    if epoch < 25:
        return 1e-1 * args.base_lr
    if epoch < 35:
        return 1e-2 * args.base_lr
    return 1e-3 * args.base_lr

# Define the function that creates the model

def cbr(x, conv_size):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(conv_size, (3,3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    return x

def conv_block(x, conv_size, scale_input = False):
    x_0 = x
    if scale_input:
        x_0 = Conv2D(conv_size, (1, 1), activation='linear', padding='same')(x_0)

    x = cbr(x, conv_size)
    x = Dropout(0.01)(x)
    x = cbr(x, conv_size)
    x = Add()([x_0, x])

    return x

def create_model():

    # Implementation of WideResNet (depth = 16, width = 10) based on keras_contrib
    # https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/wide_resnet.py

    inputs = Input(shape=(28, 28, 1))

    x = cbr(inputs, 16)

    x = conv_block(x, 160, True)
    x = conv_block(x, 160)
    x = MaxPooling2D((2, 2))(x)
    x = conv_block(x, 320, True)
    x = conv_block(x, 320)
    x = MaxPooling2D((2, 2))(x)
    x = conv_block(x, 640, True)
    x = conv_block(x, 640)
    x = GlobalAveragePooling2D()(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)

    opt = tf.keras.optimizers.SGD(lr=args.base_lr)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

verbose = 1

# Input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# Load Fashion MNIST data.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Train only on 1/6 of the dataset
x_train = x_train[:10000,:,:]
y_train = y_train[:10000]

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Training data iterator.
train_gen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                     horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2)
train_gen.fit(x_train)
train_iter = train_gen.flow(x_train, y_train, batch_size=args.batch_size)

# Validation data iterator.
test_gen = image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
test_gen.mean = train_gen.mean
test_gen.std = train_gen.std
test_iter = test_gen.flow(x_test, y_test, batch_size=args.val_batch_size)


callbacks = []
callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule))

# TODO Step 1: Define the PrintThroughput callback

# TODO Step 1: Add the PrintThrought callback to the callbacks list

# TODO Step 2: Define the StopAtAccuracy callback

# TODO Step 2: Add the StopAtAccuracy callback to the callbacks list

# TODO Step 3: Define the PrintTotalTime callback

# TODO Step 3: Add the PrintTotalTime callback to the callbacks list

# Create the model.
model = create_model()

# Train the model.
model.fit(train_iter,
          steps_per_epoch=len(train_iter),
          callbacks=callbacks,
          epochs=args.epochs,
          verbose=verbose,
          workers=4,
          initial_epoch=0,
          validation_data=test_iter,
          validation_steps=len(test_iter))

# Evaluate the model on the full data set.
score = model.evaluate(test_iter, steps=len(test_iter), workers=4)
if verbose:
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
