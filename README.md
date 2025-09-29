# Deepfake Image Detection

**Project:** Deepfake Image Detection

## Overview

This project focuses on detecting deepfake images using a pre-trained deep learning model. The implementation leverages the Kaggle 140k Deepfake dataset and the model `prithivMLmods/Deep-Fake-Detector-v2-Model`. The system allows users to run detection on new images through a simple web interface.

## Key Features

* Uses a pre-trained state-of-the-art model for deepfake detection.
* Fine-tuning support using the Kaggle 140k dataset.
* Web interface built with Flask for uploading and analyzing images.
* Easy-to-run Python scripts for both inference and fine-tuning.

## Repository Structure

```
project/
├── static/              # CSS, JS, and image assets for the web app
├── templates/           # HTML templates for Flask
├── README.md            # Project documentation
├── app.py               # Flask web app for running detection
├── fine tune.py         # Script for fine-tuning the model
├── pre-trained.py       # Script for using the pre-trained model
├── requirements.txt     # Dependencies list
```

## Getting Started

### Installation

```bash
python -m venv venv
source venv/bin/activate       # On Windows use venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Web App

```bash
python app.py
```

The Flask server will start, and you can access the application in your browser at `http://127.0.0.1:5000/`.

### Using Pre-trained Model

Run detection directly with the pre-trained model:

```bash
python pre-trained.py --image path/to/image.jpg
```

### Fine-tuning

If you want to fine-tune the model on the Kaggle 140k dataset:

```bash
python fine tune.py --data path/to/kaggle_dataset
```

## Requirements

Install all dependencies from `requirements.txt`. Typical packages include:

* torch
* torchvision
* transformers
* flask
* opencv-python
* numpy
* pandas

## Example Output

After running inference, the system will return whether an image is classified as **Real** or **Fake**, along with confidence scores.

## Ethics

This project is intended only for research and educational purposes in detecting and mitigating malicious use of deepfakes. Do not use the model or dataset for generating harmful or non-consensual content.

