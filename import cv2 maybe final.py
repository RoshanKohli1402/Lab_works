import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("sign_language_model.h5")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28)) / 255.0
    reshaped = resized.reshape(1, 28, 28, 1)

    # Predict the gesture
    prediction = model.predict(reshaped)
    predicted_label = np.argmax(prediction)

    # Display the prediction
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
