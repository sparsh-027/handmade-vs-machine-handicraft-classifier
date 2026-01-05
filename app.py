import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("handmade_vs_machine_model.h5")

def predict_handicraft(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return f"Machine-Made (Confidence: {prediction:.2f})"
    else:
        return f"Handmade (Confidence: {1 - prediction:.2f})"

# Gradio interface
interface = gr.Interface(
    fn=predict_handicraft,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Handmade vs Machine-Made Bihar Handicraft Classifier",
    description="Upload an image of a Bihar handicraft to check whether it is Handmade or Machine-Made."
)

interface.launch(share=True)