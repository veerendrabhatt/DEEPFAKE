import os
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    TFAutoModelForImageClassification
)
import evaluate

# -----------------------------
# Step 1: Load dataset
# -----------------------------
data_dir = r"D:\deepfake-detector\dataset\real-vs-fake"  # Your dataset folder
dataset = load_dataset("imagefolder", data_dir=data_dir)
print(dataset)  # Check train/validation/test splits
print("Dataset keys:", dataset.keys())  # Ensure split names

# -----------------------------
# Step 2: Load pre-trained model and processor
# -----------------------------
model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=2)

# -----------------------------
# Step 3: Preprocess dataset
# -----------------------------
def transform(example):
    # Convert single image to a list for processor
    inputs = processor(images=[example["image"]], return_tensors="pt")
    # Remove batch dimension added by processor
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    inputs["labels"] = torch.tensor(example["label"])
    return inputs

# Apply preprocessing to all splits
prepared_ds = dataset.map(transform, remove_columns=["image", "label"])

# -----------------------------
# Step 4: Metrics
# -----------------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# -----------------------------
# Step 5: Training arguments (older transformers)
# -----------------------------
training_args = TrainingArguments(
    output_dir="./fine_tuned_deepfake_model",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100
)

# -----------------------------
# Step 6: Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],  # or "valid" if that's your split key
    compute_metrics=compute_metrics
)

# -----------------------------
# Step 7: Train
# -----------------------------
trainer.train()

# -----------------------------
# Step 8: Evaluate on test set
# -----------------------------
test_results = trainer.evaluate(prepared_ds["test"])
print("Test results:", test_results)

# -----------------------------
# Step 9: Convert PyTorch model to TensorFlow/Keras
# -----------------------------
tf_model = TFAutoModelForImageClassification.from_pretrained(
    "./fine_tuned_deepfake_model",
    from_pt=True
)

# -----------------------------
# Step 10: Save as .h5
# -----------------------------
tf_model.keras_model.save("deepfake_model.h5")
print("Saved fine-tuned model as deepfake_model.h5")
