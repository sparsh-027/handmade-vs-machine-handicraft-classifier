import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ---------------------------
# 1. DATA GENERATORS
# ---------------------------

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "handicrafts/train",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary"
)
print("Class indices:", train_data.class_indices)

val_data = val_gen.flow_from_directory(
    "handicrafts/val",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    "handicrafts/test",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    shuffle=False
)

# ---------------------------
# 2. MODEL (MOBILENETV2)
# ---------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# ---------------------------
# 3. COMPILE MODEL
# ---------------------------

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------
# 4. TRAIN MODEL
# ---------------------------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


# ---------------------------
# 5. EVALUATE MODEL
# ---------------------------

loss, accuracy = model.evaluate(test_data)
print("Test Accuracy:", accuracy)

# ---------------------------
# 6. SAVE MODEL
# ---------------------------

model.save("handmade_vs_machine_model.h5")
print("Model saved successfully!")