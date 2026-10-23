from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path(__file__).parent / "models" / "chestscan_tl.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
DECISION_THRESHOLD = 0.5


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run the export cell in "
            "notebooks/chestscan_tl.ipynb first."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0), image


st.set_page_config(page_title="ChestScan", page_icon="X", layout="centered")
st.title("ChestScan")
st.caption("A MobileNetV2 chest X-ray classifier")
st.write("Upload a chest X-ray to get the model's classification.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        model = load_model()
        image_batch, image = prepare_image(uploaded_file)
        score = float(model.predict(image_batch, verbose=0)[0][0])
        prediction_index = int(score >= DECISION_THRESHOLD)

        st.image(image, caption="Uploaded X-ray", width="stretch")
        st.subheader(f"Result: {CLASS_NAMES[prediction_index].title()}")
        st.metric("Model score", f"{score:.3f}")
        st.caption(
            "This educational prototype is not a medical device and must not "
            "be used for diagnosis or clinical decisions."
        )
    except (OSError, ValueError, FileNotFoundError, tf.errors.OpError) as error:
        st.error(f"Could not process this image: {error}")