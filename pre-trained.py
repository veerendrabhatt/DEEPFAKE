# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")
pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")

import os
os.environ['HF_TOKEN'] = 'YOUR_TOKEN'

import os 
from huggingface_hub import InferenceClient 
client = InferenceClient( provider="auto", api_key=os.environ["HF_TOKEN"], ) 
output = client.image_classification("D:/deepfake-detector/dataset/real-vs-fake/test/fake/N0BA77DQ73.JPG", model="prithivMLmods/Deep-Fake-Detector-v2-Model")
print(output)

from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate

# Load dataset from local folder
data_dir = r"D:/deepfake-detector/dataset/real-vs-fake"
dataset = load_dataset("imagefolder", data_dir=data_dir)

# Load processor & model
model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=2)

# Preprocess
def transform(example):
    inputs = processor(images=example["image"], return_tensors="pt")
    inputs["labels"] = example["label"]
    return inputs

prepared_ds = dataset.with_transform(transform)
