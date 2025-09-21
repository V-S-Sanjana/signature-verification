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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Disable TensorFlow 2.x behavior to use TensorFlow 1.x style
tf.compat.v1.disable_eager_execution()

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for the model
sess = None
x = None
y = None
accuracy = None

def init_model():
    """Initialize the TensorFlow model"""
    global sess, x, y, accuracy
    
    # Load training data for model initialization
    current_dir = os.path.dirname(__file__)
    author = '021'
    training_folder = os.path.join(current_dir, 'data/training/', author)
    
    training_data = []
    training_labels = []
    for filename in os.listdir(training_folder):
        img = cv2.imread(os.path.join(training_folder, filename), 0)
        if img is not None:
            data = preprocessor.prepare(img)
            training_data.append(data)
            training_labels.append([0, 1] if "genuine" in filename else [1, 0])
    
    # Build the model
    tf.compat.v1.reset_default_graph()
    
    with tf.compat.v1.variable_scope("regression"):
        x = tf.compat.v1.placeholder(tf.float32, [None, 901])
        W = tf.compat.v1.Variable(tf.zeros([901, 2]), name="W")
        b = tf.compat.v1.Variable(tf.zeros([2]), name="b")
        y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    # Training setup
    y_ = tf.compat.v1.placeholder("float", [None, 2])
    cross_entropy = -tf.reduce_sum(y_ * tf.math.log(y + 1e-10))  # Add small epsilon for numerical stability
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Initialize session and train the model
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Train the model
    for epoch in range(100):  # More epochs for better training
        sess.run(train_step, feed_dict={x: training_data, y_: training_labels})
    
    print("Model initialized and trained successfully!")

def predict_signature(image_path):
    """Predict if a signature is genuine or forged"""
    global sess, x, y
    
    try:
        # Load and preprocess the image
        img = cv2.imread(image_path, 0)
        if img is None:
            return None, "Could not load image"
        
        # Prepare the image using the same preprocessing as training
        data = preprocessor.prepare(img)
        data_array = np.array([data])  # Add batch dimension
        
        # Make prediction
        prediction = sess.run(y, feed_dict={x: data_array})
        
        # Get the predicted class (0: forged, 1: genuine)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        if predicted_class == 1:
            result = "Genuine"
        else:
            result = "Forged"
        
        return result, confidence
    
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result, confidence = predict_signature(filepath)
        
        if result is None:
            return jsonify({'error': confidence})
        
        # Convert image to base64 for display
        with open(filepath, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2%}",
            'image': img_data
        })

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
    print("Initializing model...")
    init_model()
    print("Starting Flask application...")
    app.run(debug=True, port=5000)
