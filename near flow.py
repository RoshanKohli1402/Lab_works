import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# Step 1: Dataset 
train_dir = "dataset/train"  # Training dataset path
val_dir = "dataset/val"      # Validation dataset path

# Step 2: Data Preprocessing
datagen = ImageDataGenerator(rescale=1.0/255)

train_data = datagen.flow_from_directory(
    train_dir, target_size=(64, 64), batch_size=32, class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    val_dir, target_size=(64, 64), batch_size=32, class_mode='categorical'
)

# Step 3: Building the CNN Model 
model = Sequential
([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')  # Classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Training the Model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the trained model
model.save("sign_language_model.h5")

# Step 5: Real-time Recognition
def predict_sign(frame, model):
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    return list(train_data.class_indices.keys())[class_index], confidence

# Load the trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect hand and predict
    sign, confidence = predict_sign(frame, model)
    cv2.putText(frame, f"{sign} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()