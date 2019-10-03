from time import time
from keras import models, layers, optimizers
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

TRAIN_DIR_Apolo = "waste-classification-data/DATASET/TRAIN/"
VAL_DIR_Apolo = "waste-classification-data/DATASET/TEST/"

TRAIN_BATCHSIZE = 64
VAL_BATCHSIZE = 64

resnet_model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))

for layer in resnet_model.layers[:-1]:
    layer.trainable = False
# # print layers and thier status for training
# for layer in resnet_model.layers:
#     print(layer, layer.trainable)

# Create the model
new_model = models.Sequential()
# Add the resnet model to the new model
new_model.add(resnet_model)
# Add new layers
# new_model.add(layers.Flatten())
new_model.add(layers.Dense(2, activation = 'softmax'))

# Summary of the model
new_model.summary()

train_datagen = image.ImageDataGenerator()
val_datagen = image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR_Apolo,
    target_size=(224,224),
    batch_size=TRAIN_BATCHSIZE,
    class_mode="binary"
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR_Apolo,
    target_size=(224,224),
    batch_size=VAL_BATCHSIZE,
    class_mode="binary"
)

opt = optimizers.Adam()
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

tensorboard = TensorBoard(log_dir=f"logs/resnet/{opt}_{time}")

new_model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt,
                  metrics=["acc"])

history = new_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size,
    epochs=30,
    callbacks=[tensorboard],
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    verbose=1
)

new_model.save(f"trained_models/resnet_{opt}.h5")
