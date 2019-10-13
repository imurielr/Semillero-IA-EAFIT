from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import models, layers, optimizers
from time import time

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

train_dir = 'DATASET/TRAIN'
validation_dir = 'DATASET/TEST'

# Change the batchsize according to your system RAM
train_batchsize = 32
val_batchsize = 32

densenet = DenseNet121(weights='imagenet', input_shape=(224, 224, 3))

# Freeze the layers except the last 4 layers
for layer in densenet.layers[:-1]:
    layer.trainable = False

model = models.Sequential()
model.add(densenet)
# Add new layers
# model.add(layers.Flatten())
# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()

train_datagen = image.ImageDataGenerator()
validation_datagen = image.ImageDataGenerator()
 
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = train_batchsize,
    class_mode = 'binary'
)
 
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (224, 224),
    batch_size = val_batchsize,
    class_mode = 'binary',
)

opt = optimizers.RMSprop()

tensorboard = TensorBoard(log_dir=f"logs/{opt}_{time}")

model.compile(loss="sparse_categorical_crossentropy",
                optimizer=opt,
                metrics=["acc"])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size,
    epochs=30,
    callbacks=[tensorboard],
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    verbose=1
)

model.save(f"trained_models/densenet_{opt}.h5")
