import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Load trained model
model = tf.keras.models.load_model("handmade_vs_machine_model.h5")

# Load test data
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    "handicrafts/test",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    shuffle=False
)

# Get true labels
y_true = test_data.classes

# Get predictions
y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Handmade", "Machine-made"]))