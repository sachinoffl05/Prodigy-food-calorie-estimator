import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os

# Load trained model
model = tf.keras.models.load_model("food_model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_indices = json.load(f)

# Convert index → class name
class_names = {v: k for k, v in class_indices.items()}

# Folder containing images
folder = "../../dataset/archive/data/food-101-tiny/valid/apple_pie"

# Automatically pick the first image
img_name = os.listdir(folder)[0]
img_path = os.path.join(folder, img_name)

print("Testing image:", img_name)

# Load and prepare image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
pred_index = np.argmax(pred)
confidence = pred[0][pred_index] * 100

print("Predicted food:", class_names[pred_index])
print("Confidence:", round(confidence, 2), "%")
