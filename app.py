from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import uuid
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = 'deepfake_detector_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
IMG_SIZE = (299, 299)

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
class_names = ['Real', 'Fake']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_deepfake_model():
    """Load the trained deepfake detection model"""
    global model
    try:
        model_path = 'best_deepfake_model.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("Model loaded successfully!")
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_path):
    """Predict if image is real or fake"""
    if model is None:
        return None, 0, "Model not loaded"
    
    try:
        img_array = preprocess_image(image_path)
        if img_array is None:
            return None, 0, "Error preprocessing image"
        
        predictions = model.predict(img_array, verbose=0)
        prediction_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        class_name = class_names[prediction_class]
        
        return prediction_class, confidence, class_name
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > MAX_FILE_SIZE:
                return jsonify({'error': 'File size too large. Maximum size is 16MB.'}), 400
            
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            # Save file
            file.save(filepath)
            
            # Make prediction
            prediction_class, confidence, class_name = predict_image(filepath)
            
            if prediction_class is not None:
                # Convert image to base64 for display
                with open(filepath, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Clean up uploaded file
                os.remove(filepath)
                
                return jsonify({
                    'success': True,
                    'prediction': class_name,
                    'confidence': round(confidence, 2),
                    'image_data': img_data,
                    'filename': filename
                })
            else:
                # Clean up uploaded file
                os.remove(filepath)
                return jsonify({'error': 'Failed to process image'}), 500
                
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'upload_folder': UPLOAD_FOLDER
    })

if __name__ == '__main__':
    print("Loading deepfake detection model...")
    if load_deepfake_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please ensure the model file exists.")
        print("You can still run the app, but predictions will not work.")
        app.run(debug=True, host='0.0.0.0', port=5000)



