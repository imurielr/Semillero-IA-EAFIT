from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import models, layers, optimizers
import numpy as np

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
 
# Change the batchsize according to your system RAM
train_batchsize = 32
val_batchsize = 32

train_dir = 'DATASET/TRAIN'
validation_dir = 'DATASET/TEST'
 
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

# Compile the model
model.compile(
    loss = 'sparse_categorical_crossentropy', 
    optimizer = optimizers.RMSprop(lr=1e-4),
    metrics = ['acc']
)

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples / train_generator.batch_size,
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples / validation_generator.batch_size,
    verbose = 1
)

# Save the model
model.save('densenet.h5')

# img_path = '/Users/isamuriel/Semillero_IA_2019/DATASET/TEST/O/O_13941.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = densenet.predict(x)

# print('Predicted:', decode_predictions(preds, top=3)[0])
