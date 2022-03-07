import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import ssl

(x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(
    'Cifar10',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))
y_train = tf.one_hot(y_train,depth=10)
y_test = tf.one_hot(y_test,depth=10)


# shape of x_train ----->  (50000, 32, 32, 3)
# shape of y_train ----->  (50000, 10)











