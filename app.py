import streamlit as st
import os
import sys
import platform
import subprocess

st.title("Deepfake Detection App Simulator")

python_version = sys.version.split()[0]
arch = platform.machine()
st.info(f"Detected Environment: Python {python_version} on {arch} architecture")

if sys.version_info >= (3, 12):
    st.warning(f"⚠️ Note: You are running Python {python_version}. If you are deploying to Streamlit Cloud and face deployment errors, you may need to recreate your app using Python 3.11 in the Advanced Settings.")


@st.cache_resource
def install_heavy_packages():
    # Safe fallback if ARM64
    tf_pkg = "tensorflow-cpu==2.15.0" if arch != "aarch64" else "tensorflow-aarch64"
    out = subprocess.run(
        [sys.executable, "-m", "pip", "install", tf_pkg, "opencv-python-headless", "numpy", "--no-cache-dir"],
        capture_output=True, text=True
    )
    return out

with st.spinner("Initializing Deepfake Weights & ML Libraries (Takes ~1 minute). Do not refresh..."):
    install_result = install_heavy_packages()

if install_result.returncode != 0:
    st.error("There was a severe error installing dependencies!")
    st.code(install_result.stderr + "\n" + install_result.stdout)
    st.stop()
else:
    st.success("System initialized successfully!")

import tensorflow as tf
import cv2
import numpy as np
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

@st.cache_resource
def load_your_models():
    """Load trained weights"""
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    feature_extractor = tf.keras.Model(inputs, base_model(x))

    frame_input = tf.keras.layers.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = tf.keras.layers.Input((MAX_SEQ_LENGTH,), dtype="bool")
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(16, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))
    )(frame_input, mask=mask_input)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(8, kernel_regularizer=tf.keras.regularizers.l2(0.01))
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model([frame_input, mask_input], output)

    # Use current working directory instead of __file__
    base_dir = os.getcwd()
    weights_dir = os.path.join(base_dir, "production_weights")
    feature_path = os.path.join(weights_dir, "feature_extractor_weights.h5")
    model_path = os.path.join(weights_dir, "gru_model_weights.h5")

    if not os.path.exists(feature_path):
        st.error(f"❌ Missing {feature_path}")
    else:
        feature_extractor.load_weights(feature_path)

    if not os.path.exists(model_path):
        st.error(f"❌ Missing {model_path}")
    else:
        model.load_weights(model_path)

    model.compile("binary_crossentropy", "adam", ["accuracy"])
    return feature_extractor, model

st.title("Deepfake Detection App")
st.write("App is ready! Please upload a video to test.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.info("Model processing logic will go here. Model is loaded in the background!")
    try:
        feature_extractor, model = load_your_models()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")