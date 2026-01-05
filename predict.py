import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("handmade_vs_machine_model.h5")

# Path of image to test
img_path = "test4.jpg"   # put an image in craft_project

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Prediction: MACHINE-MADE")
else:
    print("Prediction: HANDMADE")

print("Confidence:", prediction[0][0])