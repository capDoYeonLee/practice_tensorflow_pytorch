import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU

from IPython.display import SVG
from tensorflow import keras
from keras.utils.vis_utils import model_to_dot

from time import time
import numpy as np

from pascal_voc import *


def mobilenet(input_shape, n_classes):
  
  def mobilenet_block(x, f, s=1):
    x = DepthwiseConv2D(3, strides=s, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(f, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
    
    
  input = Input(input_shape)

  x = Conv2D(32, 3, strides=2, padding='same')(input)
  x = BatchNormalization()(x)
  x = ReLU()(x)

  x = mobilenet_block(x, 64)
  x = mobilenet_block(x, 128, 2)
  x = mobilenet_block(x, 128)

  x = mobilenet_block(x, 256, 2)
  x = mobilenet_block(x, 256)

  x = mobilenet_block(x, 512, 2)
  for _ in range(5):
    x = mobilenet_block(x, 512)

  x = mobilenet_block(x, 1024, 2)
  x = mobilenet_block(x, 1024)
  
  x = GlobalAvgPool2D()(x) # TODO: look it up
  
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model, output


# input_shape = 224, 224, 3
n_classes = 20

# img, _, _ = next(iter(train_ds))
# input_shape = img.shape

# K.clear_session()
model, output_ = mobilenet(img.shape , n_classes)
model.summary()
# print(output_)


# SVG(model_to_dot(model).create(prog='dot', format='svg'))
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


initial_epochs = 10
history = model.fit(x=train_x, y=train_y,
                    epochs=initial_epochs,
                    batch_size=16
                    # validation_data=validation_dataset
                    
                    )



















