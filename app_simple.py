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
        if explanation_result["success"]:
            response_data.update({
                "explainable_ai": {
                    "heatmap_overlay": explanation_result["heatmap_overlay"],
                    "detailed_analysis": explanation_result["detailed_analysis"],
                    "region_analysis": explanation_result["region_analysis"],
                    "explanation": explanation_result["explanation"]
                }
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
    app.run(debug=True, host="0.0.0.0", port=5000)