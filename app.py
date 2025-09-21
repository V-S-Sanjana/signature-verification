from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import os
import numpy as np
import tensorflow as tf
import preprocessor
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
from model_utils import get_model_loader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Disable TensorFlow warnings
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model loader
model_loader = get_model_loader()

def predict_signature(image_data, author_id='021', model_type='both'):
    """Predict if a signature is genuine or forged using trained models"""
    try:
        results = {}
        
        if model_type in ['custom_nn', 'both']:
            # Use custom neural network
            result, error = model_loader.predict_custom_nn(image_data, author_id)
            if result:
                results['custom_nn'] = result
            else:
                results['custom_nn_error'] = error
        
        if model_type in ['tensorflow', 'both']:
            # Use TensorFlow model
            result, error = model_loader.predict_tensorflow(image_data, author_id)
            if result:
                results['tensorflow'] = result
            else:
                results['tensorflow_error'] = error
        
        return results, None
    
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Main page with model information"""
    model_info = model_loader.get_model_info()
    return render_template('index.html', model_info=model_info)

@app.route('/api/models')
def get_models():
    """API endpoint to get model information"""
    model_info = model_loader.get_model_info()
    return jsonify(model_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Get parameters
    author_id = request.form.get('author_id', '021')
    model_type = request.form.get('model_type', 'both')
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess the image
            img = cv2.imread(filepath, 0)
            if img is None:
                return jsonify({'error': 'Could not load image'})
            
            # Make prediction using both models
            results, error = predict_signature(img, author_id, model_type)
            
            if results is None:
                return jsonify({'error': error})
            
            # Convert image to base64 for display
            with open(filepath, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            response = {
                'image': img_data,
                'author_id': author_id,
                'results': results
            }
            
            return jsonify(response)
            
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)})

@app.route('/sample')
def sample():
    """Show sample images from the dataset"""
    current_dir = os.path.dirname(__file__)
    samples = []
    
    # Get some sample images from the test data
    test_folder = os.path.join(current_dir, 'data/test/021')
    if os.path.exists(test_folder):
        files = os.listdir(test_folder)
        genuine_files = [f for f in files if 'genuine' in f][:3]
        forged_files = [f for f in files if 'forged' in f][:3]
        
        for file in genuine_files:
            filepath = os.path.join(test_folder, file)
            with open(filepath, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            samples.append({
                'name': file,
                'type': 'Genuine',
                'image': img_data
            })
        
        for file in forged_files:
            filepath = os.path.join(test_folder, file)
            with open(filepath, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            samples.append({
                'name': file,
                'type': 'Forged',
                'image': img_data
            })
    
    return render_template('samples.html', samples=samples)

if __name__ == '__main__':
    print("=== Signature Verification System ===")
    print("Loading trained models...")
    
    model_info = model_loader.get_model_info()
    print(f"✓ Loaded {model_info['custom_nn_models']} custom neural network models")
    print(f"✓ Available authors: {', '.join(model_info['available_authors'])}")
    
    for author_id, details in model_info['model_details'].items():
        print(f"  - Author {author_id}:")
        if 'tensorflow_accuracy' in details:
            print(f"    TensorFlow accuracy: {details['tensorflow_accuracy']:.4f}")
        print(f"    Custom NN trained at: {details['custom_nn_trained_at']}")
    
    print("\nStarting Flask application...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
