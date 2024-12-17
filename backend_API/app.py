from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from flask_cors import CORS
from flask import Flask, request
import os

app = Flask(__name__)
CORS(app)  # Allows communication between Flutter and Flask

# Load the TensorFlow Lite model
tflite_model_path = "../saved_models/smart_waste_model.tflite"
if not os.path.exists(tflite_model_path):
    print(f"Error: The model file does not exist at {tflite_model_path}")
    exit(1)
else:
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading TensorFlow Lite model: {e}")
        exit(1)

# Class labels
labels = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}
@app.route('/')
# Preprocess function
def preprocess_image(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)  # Convert to numpy array
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode image
        if img is None:
            raise ValueError("Invalid image data")
        img = cv2.resize(img, (224, 224)) / 255.0  # Resize and normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        raise ValueError(f"Image preprocessing error: {e}")

# API Endpoint: Accepts image and returns classification result
@app.route("/classify", methods=["POST"])
def classify():
    try:
        # Check if the image file is in the request
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Retrieve the image file from the request
        image_file = request.files["image"]

        # Ensure the file is an image
        if not image_file.content_type.startswith('image'):
            return jsonify({"error": "Invalid file format. Please upload an image file."}), 400

        # Read the image bytes
        img_bytes = image_file.read()

        # Preprocess the image
        img = preprocess_image(img_bytes)

        # Run the image through the TensorFlow Lite model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor to the image data
        interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Get the results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        class_id = np.argmax(output_data)
        confidence = float(output_data[0][class_id])

        # Return the classification result
        return jsonify({
            "class_name": labels[class_id],
            "confidence": confidence
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400  # Return a 400 Bad Request for value errors
    except Exception as e:
        # Log the error and return a 500 Internal Server Error
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error. Please try again later."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
