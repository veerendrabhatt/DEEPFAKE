import io
import base64
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PIL import Image

app = Flask(__name__)

# Load Hugging Face pipeline
print("Loading model... this may take a moment.")
deepfake_detector = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")

def predict_image(image):
    """Takes a PIL image and returns prediction and confidence"""
    results = deepfake_detector(image)
    best = max(results, key=lambda x: x['score'])
    label = best['label']
    confidence = round(best['score'] * 100, 2)
    return label, confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"})

    try:
        # Read image
        image = Image.open(file.stream).convert("RGB")

        # Prediction
        label, confidence = predict_image(image)

        # Encode image as base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "image_data": img_str,
            "filename": file.filename
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
