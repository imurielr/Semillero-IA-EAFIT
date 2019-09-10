import numpy as np
from keras import models, layers, optimizers
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

TRAIN_DIR_Rafael = "../waste-classification-data/DATASET/TRAIN/"
VAL_DIR_Rafael = "../waste-classification-data/DATASET/TEST/"

TRAIN_BATCHSIZE = 64
VAL_BATCHSIZE = 64

# TRAIN_DIR_Pedro = "C:\\Users\\pedro\\OneDrive\\Documentos\\Artificial Inteligence\\DATASET"

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
    TRAIN_DIR_Rafael,
    target_size=(224,224),
    batch_size=TRAIN_BATCHSIZE,
    class_mode="binary"
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR_Rafael,
    target_size=(224,224),
    batch_size=VAL_BATCHSIZE,
    class_mode="binary"
)

new_model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=["acc"])

history = new_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    verbose=1
)

new_model.save("resnet.h5")



# # Predict with the model
# img_path = ''
# img = image.load_img(img_path, target_size=(256, 256))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = model.predict(x)

# print('Predicted:', decode_predictions(preds, top=3)[0])