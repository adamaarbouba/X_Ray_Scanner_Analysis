# ChestScan AI

# Project Description

ChestScan is a simple Streamlit app that uses a TensorFlow/Keras MobileNetV2
model to classify uploaded chest X-ray images as `NORMAL` or `PNEUMONIA`.

The model is trained in `notebooks/chestscan_tl.ipynb` and exported as
`models/chestscan_tl.keras`. The app resizes each uploaded image to `224 x 224`,
converts it to RGB, and then passes it to the exported model.

## Project Files

- `app.py`: Streamlit application
- `models/chestscan_tl.keras`: exported trained model
- `notebooks/chestscan_tl.ipynb`: training and evaluation notebook
- `requirements.txt`: Python dependencies

## Run the App

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit from the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal and upload a JPG or PNG chest
X-ray.

## Train or Export the Model

The notebook downloads the Chest X-Ray Images (Pneumonia) dataset from Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Run the notebook through training, then run its final export cell. The export
cell creates `models/chestscan_tl.keras`, which is the file used by the app.

## Disclaimer

This is an educational project, not a medical device. Do not use its results
for diagnosis or clinical decisions.

## Disclaimer

This educational prototype is not a medical device and must not be used for medical diagnosis or clinical decisions.
