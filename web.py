import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import traceback

# Load trained Keras model
print("✅ Loading model...")
try:
    model = load_model("deepfake_detector.h5")
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Find the last convolutional layer dynamically
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

last_conv_layer_name = get_last_conv_layer(model)
if not last_conv_layer_name:
    raise ValueError("No convolutional layer found in the model!")
print(f"✅ Using last conv layer: {last_conv_layer_name}")

# Define preprocessing
def preprocess_image(image_pil):
    try:
        image_pil = image_pil.resize((256, 256))  # Ensure size matches model input
        image_array = np.array(image_pil)
        if image_array.shape[-1] == 4:  # Handle RGBA images
            image_array = image_array[:, :, :3]
        image_array = preprocess_input(image_array)  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        traceback.print_exc()
        raise

# Grad-CAM function
def get_gradcam_heatmap(model, image_array, target_class):
    try:
        grad_model = tf.keras.models.Model(
            [model.input], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array, training=False)
            # Handle both single-value and two-class output formats
            if predictions.shape[-1] == 1:
                loss = predictions[:, 0]  # Single output
            else:
                loss = predictions[:, target_class]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        heatmap = np.zeros(dtype=np.float32, shape=conv_outputs.shape[:-1])

        for i in range(pooled_grads.shape[-1]):
            heatmap += pooled_grads[i] * conv_outputs[:, :, i]

        heatmap = np.maximum(heatmap, 0)  # ReLU operation

        if np.max(heatmap) > 0:
            # Fixed normalization line:
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)  # Avoid division by zero

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (256, 256))  # Resize to match input dimensions
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB

        return heatmap
    except Exception as e:
        print(f"❌ Error generating Grad-CAM: {e}")
        traceback.print_exc()
        # Return a blank red image to indicate error
        blank_heatmap = np.zeros((256, 256, 3), dtype=np.uint8)
        blank_heatmap[:, :, 0] = 255  # Red color to indicate error
        return blank_heatmap

# Prediction function
def predict(image_np):
    if image_np is None:
        return "No image provided", {"Error": 1.0}, np.zeros((256, 256, 3), dtype=np.uint8)
    
    print("🔄 Received image for prediction...")
    try:
        image_pil = Image.fromarray(image_np)
        input_tensor = preprocess_image(image_pil)
        
        print("🔄 Running model inference...")
        predictions = model.predict(input_tensor, verbose=0)  # Silent inference
        
        # Determine output format and handle accordingly
        if predictions.shape[-1] == 1:  # Single output (binary classification with sigmoid)
            print("Detected single-output model")
            prediction_value = float(predictions[0][0])
            
            # REVERSED LOGIC: In this model, output value meaning is reversed
            # High value = Real, Low value = Fake (based on your feedback)
            if prediction_value >= 0.5:
                # For this model, high values mean Real
                predicted_label = "Real"
                real_confidence = prediction_value
                fake_confidence = 1 - prediction_value
            else:
                # For this model, low values mean Fake
                predicted_label = "Fake"
                real_confidence = prediction_value
                fake_confidence = 1 - prediction_value
            
            # Debug prints to validate predictions
            print(f"Raw prediction value: {prediction_value}")
            print(f"Interpreted as: {predicted_label}")
            print(f"Real confidence: {real_confidence}, Fake confidence: {fake_confidence}")
            
            # Display confidence scores properly
            confidence = np.array([real_confidence, fake_confidence])
            class_labels = ["Real", "Fake"]
            
            # For Grad-CAM target
            target_class = 0  # Focus on what the model sees as "real" features
        else:  # Two outputs (softmax)
            print("Detected two-output model")
            confidence = tf.nn.softmax(predictions[0]).numpy()
            class_labels = ["Real", "Fake"]
            predicted_label = class_labels[np.argmax(confidence)]
            target_class = np.argmax(confidence)
        
        print(f"✅ Prediction complete! Result: {predicted_label} with confidence {confidence}")
        
        print("🔄 Generating Grad-CAM heatmap...")
        heatmap = get_gradcam_heatmap(model, input_tensor, target_class=target_class)
        print("✅ Grad-CAM generated!")

        # Create confidence dictionary
        confidence_dict = {}
        for i, label in enumerate(class_labels):
            if i < len(confidence):
                confidence_dict[label] = float(confidence[i])
            else:
                confidence_dict[label] = 0.0
        
        # ADD OVERRIDE FOR TESTING - Remove this block once working correctly
        # If fake_confidence > real_confidence, override the prediction regardless of threshold
        if fake_confidence > real_confidence:
            predicted_label = "Fake"
            print("⚠️ Override applied: Fake confidence higher than Real, setting label to Fake")

        return predicted_label, confidence_dict, heatmap
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", {"Error": 1.0}, np.zeros((256, 256, 3), dtype=np.uint8)

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.JSON(label="Confidence Scores"),
        gr.Image(type="numpy", label="Grad-CAM Heatmap")
    ],
    title="AI-Based Deepfake Detector",
    description="Upload an image to check if it's real or fake. The model provides explainability using Grad-CAM heatmaps.",
    examples=[
        # Optional: Add example images if you have them
    ]
)

if __name__ == "__main__":
    print("🚀 Starting Gradio app...")
    demo.launch(share=True, debug=True)