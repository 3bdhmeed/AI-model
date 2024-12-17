from tensorflow.keras.models import load_model
import cv2
import numpy as np
from data_preprocessing import get_data_generators

# Load the trained model
model_path = '../saved_models/smart_waste_model.keras'
model = load_model(model_path)

# Load labels
_, validation_generator = get_data_generators()
labels = list(validation_generator.class_indices.keys())

# Function to classify an image frame
def classify_image(frame, model, labels, threshold=0.6):
    img = cv2.resize(frame, (224, 224))  # Resize to model input size
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize and add batch dim
    predictions = model.predict(img, verbose=0)
    max_prob = np.max(predictions)
    if max_prob < threshold:
        return "Uncertain", None
    class_id = np.argmax(predictions)
    return labels[class_id], max_prob

# Initialize webcam for real-time classification
def real_time_classification(model, labels, threshold=0.6):
    cap = cv2.VideoCapture(0)  # 0 indicates the default webcam

    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return

    print("Starting Real-Time Waste Classification... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Classify the current frame
        label, confidence = classify_image(frame, model, labels, threshold)

        # Overlay prediction results on the frame
        text = f"{label} ({confidence:.2f})" if confidence else label
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with predictions
        cv2.imshow("Real-Time Waste Classification", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Real-Time Classification Stopped.")

# Main function to run the script
if __name__ == "__main__":
    real_time_classification(model, labels, threshold=0.6)
    # Evaluate the model performance on the validation set
    print("Evaluating model on validation set...")
    _, validation_generator = get_data_generators()
    validation_loss, validation_accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {validation_loss}")
    print(f"Validation Accuracy: {validation_accuracy}")
