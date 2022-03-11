import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from IPython.display import SVG
from tensorflow import keras
from keras.utils.vis_utils import model_to_dot

from time import time
import numpy as np
from cifar10_dataset import *



def mobilenetv1(x,alph = 1): # 224 224 3
    def dw(x,dw_pad,conv_f,conv_st):
            x = DepthwiseConv2D(kernel_size=(3,3),padding = 'same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x) 
            x = Conv2D(filters= conv_f,kernel_size=(1,1),strides=conv_st,padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
    x = Conv2D(filters=int(32*alph),kernel_size=(3,3),strides=2,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = dw(x,'same',int(64*alph),1)
    x = dw(x,'valid',int(128*alph),2)
    x = dw(x,'same',int(128*alph),1)
    x = dw(x,'same',int(256*alph),2)
    x = dw(x,'same',int(256*alph),1)
    x = dw(x,'valid',int(512*alph),2)
    for i in range(5):
        x = dw(x,'same',int(512*alph),1) 
    x = dw(x,'valid',int(1024*alph),2)
    x = dw(x,'same',int(1024*alph),1)
    return x


