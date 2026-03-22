import streamlit as st
import cv2
import numpy as np
import os
import tensorflow as tf

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
MODEL_PATH = "production_weights/gru_model_complete.h5"

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==============================
# FEATURE EXTRACTOR
# ==============================
@st.cache_resource
def build_feature_extractor():
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    outputs = base_model(x)
    return tf.keras.Model(inputs, outputs)

feature_extractor = build_feature_extractor()

# ==============================
# VIDEO PROCESSING
# ==============================
def square_crop_frame(image):
    h, w = image.shape[:2]
    size = min(h, w)
    start_x, start_y = (w - size) // 2, (h - size) // 2
    return image[start_y:start_y+size, start_x:start_x+size]

def process_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    for _ in range(MAX_SEQ_LENGTH):
        ret, frame = cap.read()
        if not ret:
            break
        frame = square_crop_frame(frame)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    orig_len = len(frames)
    if orig_len == 0:
        raise ValueError("No frames extracted from video.")

    if orig_len < MAX_SEQ_LENGTH:
        pad_frames = np.repeat(frames[-1:], MAX_SEQ_LENGTH - orig_len, axis=0)
        frames += list(pad_frames)
    else:
        frames = frames[:MAX_SEQ_LENGTH]

    mask = np.zeros((MAX_SEQ_LENGTH,), dtype=bool)
    mask[:orig_len] = True

    return np.array(frames), mask

# ==============================
# UI
# ==============================
st.title("🎥 Deepfake Detector")
st.write("Upload a video and detect whether it is REAL or FAKE.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing video..."):
        try:
            # Process video frames
            frames, mask = process_video_frames(temp_video_path)

            # Extract features (batch)
            features = feature_extractor.predict(frames, verbose=0)

            # Prepare input for model
            features = np.expand_dims(features, axis=0)
            mask = np.expand_dims(mask, axis=0)

            # Predict
            prediction = model.predict([features, mask], verbose=0)[0][0]

            # Debug: raw prediction
            st.write("Raw prediction:", float(prediction))

            # Determine label and confidence
            if prediction > 0.5:
                label = "FAKE"
                confidence = prediction
            else:
                label = "REAL"
                confidence = 1 - prediction

            # Display result
            if label == "FAKE":
                st.error(f"🚨 Prediction: FAKE (Confidence: {confidence:.2f})")
            else:
                st.success(f"✅ Prediction: REAL (Confidence: {confidence:.2f})")

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)