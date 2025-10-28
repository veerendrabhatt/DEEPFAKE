import os

# Force headless matplotlib backend (must be set before any plotting imports)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

import io
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image
from simple_explainer import SimpleDeepfakeExplainer
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Initialize simple explainable AI system
print("Loading Simple Deepfake Detection with Explainable AI...")
try:
    explainer = SimpleDeepfakeExplainer()
    print("✓ Simple Explainable AI system ready!")
except Exception as e:
    print(f"❌ Error initializing explainer: {e}")
    explainer = None

def predict_image(image):
    """Takes a PIL image and returns prediction and confidence"""
    if explainer is None:
        return "Error", 0.0
    
    try:
        prediction, confidence = explainer.simple_predict(image)
        return prediction, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0

def _normalize_score(v):
    try:
        v = float(v)
    except Exception:
        return 0.0
    # if percentages were returned (0-100) convert to 0-1
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))

def compute_label_confidence_from_regions(region_scores=None, most_suspicious=None, threshold=0.5):
    """
    Derive label and confidence from region-level scores.
    - region_scores: dict of region -> score (0..1 or 0..100)
    - most_suspicious: optional tuple/name returned by explainer
    Returns (label:str, confidence:float, used_region:str, score:float)
    """
    if not region_scores:
        # fallback to most_suspicious if provided
        if most_suspicious:
            name = most_suspicious if isinstance(most_suspicious, str) else str(most_suspicious)
            # unknown numeric score, fallback default
            return ("Fake" if threshold <= 0.5 else "Real", round(0.0, 2), name, 0.0)
        return ("Unknown", 0.0, None, 0.0)

    # normalize values and find max
    norm = {k: _normalize_score(v) for k, v in region_scores.items()}
    used_region = max(norm, key=lambda k: norm[k])
    score = norm.get(used_region, 0.0)
    label = "Fake" if score >= threshold else "Real"
    confidence = round(score * 100.0, 2)
    return label, confidence, used_region, score

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

        # Basic prediction
        label, confidence = predict_image(image)

        # Generate explainable AI analysis
        if explainer:
            explanation_result = explainer.explain_prediction(image)
        else:
            explanation_result = {"success": False, "error": "Explainer not available"}

        # Encode original image as base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response_data = {
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "image_data": img_str,
            "filename": file.filename
        }

        # Add explainable AI data if successful
        if explanation_result.get("success"):
            # Compute derived prediction/confidence from region scores when available
            region_scores = explanation_result.get("region_analysis", {}).get("region_scores") or explanation_result.get("region_scores")
            most_suspicious = explanation_result.get("most_suspicious") or explanation_result.get("most_suspicious_region")
            derived_label, derived_conf, used_region, used_score = compute_label_confidence_from_regions(region_scores, most_suspicious)

            # If the explainer provides region-based signal, override basic label/confidence
            if region_scores or most_suspicious:
                response_data["prediction"] = derived_label
                response_data["confidence"] = derived_conf
                # include metadata for UI
                response_data["most_suspicious"] = used_region
                response_data["most_suspicious_score"] = round(used_score * 100, 2)

            # Provide explainable_ai object for backward compatibility with the UI
            response_data.update({
                "explainable_ai": {
                    "heatmap_overlay": explanation_result.get("heatmap_overlay"),
                    "detailed_analysis": explanation_result.get("detailed_analysis"),
                    "region_analysis": explanation_result.get("region_analysis"),
                    "explanation": explanation_result.get("explanation")
                },
                "explanation": explanation_result
            })
        else:
            response_data["explainable_ai_error"] = explanation_result.get("error", "Unknown error")

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/explain", methods=["POST"])
def explain_only():
    """Endpoint specifically for generating explanations"""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"})

    try:
        # Read image
        image = Image.open(file.stream).convert("RGB")
        
        # Generate comprehensive explanation
        if explainer:
            explanation_result = explainer.explain_prediction(image)
        else:
            explanation_result = {"success": False, "error": "Explainer not available"}
        
        if explanation_result["success"]:
            # compute derived label/confidence for explain endpoint as well
            region_scores = explanation_result.get("region_analysis", {}).get("region_scores") or explanation_result.get("region_scores")
            most_suspicious = explanation_result.get("most_suspicious") or explanation_result.get("most_suspicious_region")
            derived_label, derived_conf, used_region, used_score = compute_label_confidence_from_regions(region_scores, most_suspicious)
            explanation_result["derived_label"] = derived_label
            explanation_result["derived_confidence"] = derived_conf
            explanation_result["most_suspicious"] = used_region
            explanation_result["most_suspicious_score"] = round(used_score * 100, 2)

            return jsonify({
                "success": True,
                "explanation": explanation_result
            })
        else:
            return jsonify({
                "success": False,
                "error": explanation_result.get("error", "Failed to generate explanation")
            })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 DEEPFAKE DETECTION WITH EXPLAINABLE AI")
    print("="*60)
    print("✓ Simple AI-powered deepfake detection")
    print("✓ Facial region analysis")
    print("✓ Visual heatmap explanations") 
    print("✓ Interactive web interface")
    print("="*60)
    print("🌐 Starting Flask server...")
    print("   Open your browser to: http://localhost:5000")
    print("="*60)
    # Run without reloader and with debug off to avoid background threads creating GUI windows
    app.run(debug=False, host="0.0.0.0", port=5002, use_reloader=False)
