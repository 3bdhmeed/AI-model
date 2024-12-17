from tensorflow.keras.models import load_model
from classify_waste import classify_image
import cv2



# Load the trained model and labels
model = load_model("./saved_models/smart_waste_model.keras")
labels = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform classification
    class_name, confidence = classify_image(frame, model, labels)
    if confidence:
        cv2.putText(frame, f"Class: {class_name} ({confidence*100:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Waste Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
