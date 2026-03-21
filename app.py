import streamlit as st
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