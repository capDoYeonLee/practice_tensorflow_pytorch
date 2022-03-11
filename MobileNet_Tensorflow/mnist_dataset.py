import tensorflow as tf




class Compose(object):
    def __init__(self, my_transforms):
        self.transforms = my_transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img




# Download the mnist dataset using keras
data_train, data_test = tf.keras.datasets.mnist.load_data()

# Parse images and labels
(train_images, train_labels) = data_train
(test_images, test_labels) = data_test


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0
test_images = test_images / 255.0


