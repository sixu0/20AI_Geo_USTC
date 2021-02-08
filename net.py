import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights=None, input_size1=(None, None, 1)):
    input = Input(input_size1, name='input')
    conv1 = Conv2D(16, (3,3), activation='relu', padding='same')(input)
    conv1 = Conv2D(16, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(64, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(128, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3,3), activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling2D(size=(2,2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(64, (3,3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, (3,3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(32, (3,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(32, (3,3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2,2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(16, (3,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(16, (3,3), activation='relu', padding='same')(conv7)

    conv8 = Conv2D(1, (1,1), activation='sigmoid')(conv7)
    model = Model(inputs=[input], outputs=[conv8])
    model.summary()
    return model
