import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf


# Load the trained model
model = tf.keras.models.load_model("cnn_bilstm_raw.keras")

@tf.function(reduce_retracing=True)
def predict_batch(images):
    return model(images)

def preprocess_image(image, bbox):
    x, y, w, h = bbox
    digit_img = image[max(0, y):y + h, max(0, x):x + w]
    digit_img = cv2.resize(digit_img, (28, 28))
    digit_img = digit_img.astype("float32") / 255.0
    digit_img = np.expand_dims(digit_img, axis=(0, -1))  # (1, 28, 28, 1)
    return digit_img

def predict_digits(image):
    img = np.array(image.convert('L'))  # Convert to grayscale
    x, y, w, h = 253, 459, 20, 16
    cropped_img = img[y:y + h, x:x + w]
    enlarged_img = cv2.resize(cropped_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    adaptive_thresh = cv2.adaptiveThreshold(enlarged_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    morph = cv2.erode(morph, kernel, iterations=1)
    morph = cv2.medianBlur(morph, 3)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x_c, y_c, w_c, h_c = map(int, cv2.boundingRect(contour))
        if w_c > 10 and h_c > 15:
            if w_c > 25:
                bounding_boxes.append((x_c, y_c, w_c // 2, h_c))
                bounding_boxes.append((x_c + w_c // 2, y_c, w_c // 2, h_c))
            else:
                bounding_boxes.append((x_c - 5, y_c - 5, w_c + 10, h_c + 10))
    
    if not bounding_boxes:
        return "No digits detected."

    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    digit_images = np.vstack([preprocess_image(enlarged_img, bbox) for bbox in bounding_boxes])
    digit_images = digit_images.reshape(-1, 28, 28, 1)
    digit_images = tf.convert_to_tensor(digit_images, dtype=tf.float32)
    predictions = predict_batch(digit_images)
    digit_predictions = [str(np.argmax(pred)) for pred in predictions.numpy()]
    
    return "".join(digit_predictions)

# Streamlit UI
st.title("Digit Recognition App")
st.write("Upload an image to predict the digit sequence.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed parameter
    st.write("Processing...")

    try:
        result = predict_digits(image)
        st.write(f"**Predicted Digits:** {result}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
