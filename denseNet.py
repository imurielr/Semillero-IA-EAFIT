from keras.applications.densenet import DenseNet201, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

model = DenseNet201(weights="imagenet")

img_path = ''
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])