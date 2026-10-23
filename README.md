<!-- Skeleton documentation for the ChestScan AI project. -->
# ChestScan AI

## Project Description

TODO: Describe the prototype that classifies chest X-rays as `NORMAL` or `PNEUMONIA` with a TensorFlow/Keras CNN.

## Dataset Source

TODO: Document the Chest X-Ray Images (Pneumonia) dataset by Kermany et al., *Cell* (2018), licensed under CC BY 4.0.

Dataset link: [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset must be downloaded with `kagglehub` and must not be stored in this repository.

## Preprocessing

The transfer-learning model in `notebooks/chestscan_tl.ipynb` accepts RGB images resized to `224 x 224`. The exported Keras model includes the model-side rescaling layer that maps pixel values from `[0, 255]` to `[-1, 1]`. The Streamlit app applies the same RGB conversion and resize when an image is uploaded.

## CNN Architecture

TODO: Describe the CNN layers and output configuration.

## Results

TODO: Record evaluation metrics and relevant plots after training.

## Limits and Improvements

TODO: Discuss dataset limitations, generalization, calibration, explainability, and future improvements.

## Project Structure

TODO: Describe the notebook, source modules, app, models, and documentation directories.

## Installation and Data Download

TODO: Document Python 3.10+ setup, dependency installation, Kaggle authentication, and `kagglehub` dataset download steps.

## How to Run the App

1. Install the dependencies with `pip install -r requirements.txt`.
2. Run `notebooks/chestscan_tl.ipynb` through the model training and threshold cells, then run its final export cell. This creates `models/chestscan_tl.keras`.
3. Start the app from the project root:

	```bash
	streamlit run app.py
	```

The app uses the exported model and does not download or require the training dataset at inference time.

## Disclaimer

This educational prototype is not a medical device and must not be used for medical diagnosis or clinical decisions.
