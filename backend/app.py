from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
from io import BytesIO

app = Flask(__name__)

# -------------------------------
# Load trained model
# -------------------------------
model = tf.keras.models.load_model("model/food_model.keras")

# -------------------------------
# Load class names
# -------------------------------
with open("model/class_names.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> food name
class_names = {v: k for k, v in class_indices.items()}

# -------------------------------
# Calorie data (average per serving)
# -------------------------------
calorie_data = {
    "apple_pie": 300,
    "bibimbap": 550,
    "cannoli": 220,
    "edamame": 120,
    "falafel": 330,
    "french_toast": 350,
    "ice_cream": 210,
    "ramen": 450,
    "sushi": 300,
    "tiramisu": 420
}

# -------------------------------
# Home route (Frontend)
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# Prediction route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    # Convert uploaded file to BytesIO
    img_bytes = BytesIO(file.read())

    # Load and preprocess image
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    confidence = round(float(preds[0][idx]) * 100, 2)

    food = class_names[idx]
    calories = calorie_data.get(food, "N/A")

    return jsonify({
        "food": food,
        "confidence": confidence,
        "calories": calories
    })


# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
