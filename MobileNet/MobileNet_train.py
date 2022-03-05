from model import *
from pascal_voc import *
import tensorflow as tf

train_x = img





base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])



model = mobilenet(img.shape , n_classes)
model.summary()

initial_epochs = 10
history = model.fit(#미정,
                    epochs=initial_epochs,
                    # validation_data=validation_dataset
                    )
                    










