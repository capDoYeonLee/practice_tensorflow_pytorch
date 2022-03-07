from model import *
from cifar10_dataset import *
import tensorflow as tf




filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(30, 128)
checkpoint = ModelCheckpoint(filename,monitor='val_loss', verbose=1,save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_loss',patience=10)
reduceLR = ReduceLROnPlateau( monitor='val_loss',factor=0.5,patience=3,)


inputs = Input(shape = (32,32,3),dtype=np.float32)
x = mobilenetv1(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = Dense(10, activation='softmax')(x)
model = tf.keras.models.Model(inputs,outputs)
nadam = tf.keras.optimizers.Nadam(lr=0.01)
model.compile(optimizer=nadam,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=128, epochs=30,validation_split=0.1,callbacks=[reduceLR,checkpoint,earlystopping])










