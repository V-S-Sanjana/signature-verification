#!/usr/bin/env python3
"""
Comprehensive training script for signature verification models.
This script trains both the custom neural network and TensorFlow models.
"""

import cv2
import os
import numpy as np
import tensorflow as tf
import pickle
import json
from datetime import datetime
import network
import preprocessor

# Disable TensorFlow warnings for cleaner output
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data_for_author(author_id):
    """Load training and test data for a specific author."""
    current_dir = os.path.dirname(__file__)
    training_folder = os.path.join(current_dir, 'data/training/', author_id)
    test_folder = os.path.join(current_dir, 'data/test/', author_id)
    
    print(f"Loading data for author {author_id}...")
    
    # Load training data
    training_data = []
    training_labels = []
    
    if os.path.exists(training_folder):
        for filename in os.listdir(training_folder):
            img_path = os.path.join(training_folder, filename)
            img = cv2.imread(img_path, 0)
            if img is not None:
                data = np.array(preprocessor.prepare(img))
                training_data.append(data)
                # Label: 1 for genuine, 0 for forged
                label = 1 if "genuine" in filename else 0
                training_labels.append(label)
    
    # Load test data
    test_data = []
    test_labels = []
    
    if os.path.exists(test_folder):
        for filename in os.listdir(test_folder):
            img_path = os.path.join(test_folder, filename)
            img = cv2.imread(img_path, 0)
            if img is not None:
                data = np.array(preprocessor.prepare(img))
                test_data.append(data)
                label = 1 if "genuine" in filename else 0
                test_labels.append(label)
    
    print(f"Loaded {len(training_data)} training samples and {len(test_data)} test samples")
    return training_data, training_labels, test_data, test_labels


def train_custom_neural_network(training_data, training_labels, test_data, test_labels, author_id):
    """Train the custom neural network model."""
    print(f"\n=== Training Custom Neural Network for Author {author_id} ===")
    
    # Prepare data in the format expected by the custom network
    train_formatted = []
    for data, label in zip(training_data, training_labels):
        data_reshaped = np.reshape(data, (901, 1))
        result = np.array([[0], [1]]) if label == 1 else np.array([[1], [0]])
        train_formatted.append((data_reshaped, result))
    
    test_formatted = []
    for data, label in zip(test_data, test_labels):
        data_reshaped = np.reshape(data, (901, 1))
        test_formatted.append((data_reshaped, label))
    
    # Create and train the network
    net = network.NeuralNetwork([901, 500, 500, 2])
    print("Training network...")
    net.sgd(train_formatted, epochs=30, batch_size=10, alpha=0.1, test_data=test_formatted)
    
    # Save the trained model
    model_path = f'models/custom_nn_author_{author_id}.pkl'
    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'weights': net.weights,
            'biases': net.biases,
            'sizes': net.sizes,
            'author_id': author_id,
            'trained_at': datetime.now().isoformat()
        }, f)
    
    print(f"Custom neural network saved to {model_path}")
    return net


def train_tensorflow_model(training_data, training_labels, test_data, test_labels, author_id):
    """Train the TensorFlow model."""
    print(f"\n=== Training TensorFlow Model for Author {author_id} ===")
    
    # Convert labels to one-hot encoding
    train_labels_onehot = [[1, 0] if label == 0 else [0, 1] for label in training_labels]
    test_labels_onehot = [[1, 0] if label == 0 else [0, 1] for label in test_labels]
    
    # Convert to numpy arrays
    X_train = np.array(training_data, dtype=np.float32)
    y_train = np.array(train_labels_onehot, dtype=np.float32)
    X_test = np.array(test_data, dtype=np.float32)
    y_test = np.array(test_labels_onehot, dtype=np.float32)
    
    # Reset the default graph
    tf.compat.v1.reset_default_graph()
    
    # Define the model
    with tf.compat.v1.variable_scope("regression"):
        x = tf.compat.v1.placeholder(tf.float32, [None, 901], name="input")
        W = tf.compat.v1.Variable(tf.zeros([901, 2]), name="weights")
        b = tf.compat.v1.Variable(tf.zeros([2]), name="biases")
        y = tf.nn.softmax(tf.matmul(x, W) + b, name="output")
    
    # Define loss and optimizer
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 2], name="labels")
    cross_entropy = -tf.reduce_sum(y_ * tf.math.log(tf.clip_by_value(y, 1e-10, 1.0)))
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Training session
    saver = tf.compat.v1.train.Saver()
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        print("Training TensorFlow model...")
        for epoch in range(100):
            sess.run(train_step, feed_dict={x: X_train, y_: y_train})
            
            if epoch % 20 == 0:
                train_acc = sess.run(accuracy, feed_dict={x: X_train, y_: y_train})
                test_acc = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})
                print(f"Epoch {epoch}: Train accuracy = {train_acc:.4f}, Test accuracy = {test_acc:.4f}")
        
        # Final accuracy
        final_accuracy = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})
        print(f"Final test accuracy: {final_accuracy:.4f}")
        
        # Save the model
        model_dir = f'models/tensorflow_author_{author_id}'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.ckpt')
        save_path = saver.save(sess, model_path)
        
        # Save model metadata
        metadata = {
            'author_id': author_id,
            'final_accuracy': float(final_accuracy),
            'trained_at': datetime.now().isoformat(),
            'model_path': save_path
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"TensorFlow model saved to {model_dir}")
        return final_accuracy


def main():
    """Main training function."""
    print("=== Signature Verification Model Training ===")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Available authors in the dataset
    current_dir = os.path.dirname(__file__)
    training_dir = os.path.join(current_dir, 'data/training')
    authors = [d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))]
    
    print(f"Found authors: {authors}")
    
    results = {}
    
    for author_id in authors:
        print(f"\n{'='*60}")
        print(f"Processing Author {author_id}")
        print(f"{'='*60}")
        
        # Load data
        training_data, training_labels, test_data, test_labels = load_data_for_author(author_id)
        
        if len(training_data) == 0 or len(test_data) == 0:
            print(f"No data found for author {author_id}, skipping...")
            continue
        
        # Train custom neural network
        custom_net = train_custom_neural_network(
            training_data, training_labels, test_data, test_labels, author_id
        )
        
        # Train TensorFlow model
        tf_accuracy = train_tensorflow_model(
            training_data, training_labels, test_data, test_labels, author_id
        )
        
        results[author_id] = {
            'tensorflow_accuracy': tf_accuracy,
            'custom_nn_trained': True
        }
    
    # Save training summary
    summary = {
        'training_completed_at': datetime.now().isoformat(),
        'results': results,
        'opencv_version': cv2.__version__,
        'tensorflow_version': tf.__version__
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print("Summary:")
    for author_id, result in results.items():
        print(f"Author {author_id}: TF accuracy = {result['tensorflow_accuracy']:.4f}")
    
    print(f"\nAll models saved in the 'models' directory")
    print(f"Training summary saved to 'models/training_summary.json'")


if __name__ == '__main__':
    main()
