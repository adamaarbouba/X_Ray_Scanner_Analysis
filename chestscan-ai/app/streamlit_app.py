# Streamlit interface for the ChestScan AI educational prototype.

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.predict import predict_xray


# TODO: Replace the placeholder page title with the final product title.
st.title("ChestScan AI")
st.warning("Educational prototype only. This tool is not for medical diagnosis.")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpeg", "jpg", "png", "bmp"],
)

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Uploaded chest X-ray")

        if st.button("Predict"):
            try:
                label, pneumonia_score = predict_xray(image_bytes)
                st.write(f"Predicted class: {label}")
                st.write(f"PNEUMONIA score: {pneumonia_score:.4f}")
            except Exception as error:
                st.error(f"Prediction is unavailable: {error}")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Invalid image file: {error}")
