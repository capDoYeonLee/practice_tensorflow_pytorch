from model import *
from pascal_voc import *
import tensorflow as tf






base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])



initial_epochs = 10
history = model.fit(x=train_x, y=train_y,
                    epochs=initial_epochs,
                    # validation_data=validation_dataset
                    batch_size=32
                    )
                    










