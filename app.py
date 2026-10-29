
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "pneumonia_densenet121.keras",
        compile=False
    )

st.title("Pneumonia Detection Using Chest X-Rays")
st.write("Upload a chest X-ray to predict the class and view Grad-CAM.")

try:
    model = load_model()
    base_model = model.get_layer("densenet121")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


def make_gradcam(original, model_input):
    # Use the original, unprocessed image for the overlay.
    # Use DenseNet-preprocessed pixels only for model calculations.
    last_conv = base_model.get_layer("conv5_block16_concat")

    feature_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[last_conv.output, base_model.output]
    )

    pool_layer = model.layers[-3]
    dropout_layer = model.layers[-2]
    classifier_layer = model.layers[-1]

    with tf.GradientTape() as tape:
        conv_outputs, final_features = feature_model(
            model_input, training=False
        )
        pooled = pool_layer(final_features)
        dropped = dropout_layer(pooled, training=False)
        prediction = classifier_layer(dropped)
        score = prediction[:, 0]

    gradients = tape.gradient(score, conv_outputs)
    if gradients is None:
        raise ValueError("Grad-CAM gradients are unavailable.")

    weights = tf.reduce_mean(
        gradients, axis=(1, 2), keepdims=True
    )
    heatmap = tf.reduce_sum(
        weights * conv_outputs, axis=-1
    )[0]
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return overlay


uploaded_file = st.file_uploader(
    "Choose a chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=350)

    if st.button("Predict"):
        resized = image.resize((224, 224))
        original = np.asarray(resized, dtype=np.uint8)

        array = original.astype(np.float32)[None, ...]
        model_input = tf.keras.applications.densenet.preprocess_input(
            array.copy()
        )

        with st.spinner("Analyzing X-ray..."):
            try:
                score = float(
                    model.predict(model_input, verbose=0)[0][0]
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Prediction")
                    if score >= 0.9:
                        st.error("Prediction: PNEUMONIA")
                        st.write(f"Pneumonia score: {score:.2%}")
                    else:
                        st.success("Prediction: NORMAL")
                        st.write(f"Normal score: {(1-score):.2%}")

                    st.caption(
                        "This score is the model output, not a verified "
                        "medical probability."
                    )

                with col2:
                    st.subheader("Grad-CAM Heatmap")
                    overlay = make_gradcam(original, model_input)
                    st.image(
                        overlay,
                        caption="Regions influencing the model",
                        width=350
                    )

            except Exception as e:
                st.error(f"Prediction or Grad-CAM failed: {e}")

st.warning(
    "Educational project only. This is not a medical diagnosis."
)
