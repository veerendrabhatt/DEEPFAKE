import os
import io
import base64
from flask import Flask, render_template, request
from transformers import pipeline
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load Hugging Face pipeline
print("Loading model... this may take a moment.")
deepfake_detector = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")

def predict_image(image):
    """
    Takes a PIL image and returns prediction and confidence
    """
    results = deepfake_detector(image)
    # results is a list of dicts with 'label' and 'score'
    best = max(results, key=lambda x: x['score'])
    label = best['label']
    confidence = round(best['score'] * 100, 2)
    return label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected.")

        try:
            # Read image
            image = Image.open(file.stream).convert("RGB")

            # Prediction
            label, confidence = predict_image(image)

            # Convert image to base64 for display
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return render_template(
                "index.html",
                prediction=f"{label} ({confidence}%)",
                uploaded_image=img_str,
            )
        except Exception as e:
            return render_template("index.html", prediction=f"Error: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
