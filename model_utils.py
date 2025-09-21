"""
Model loading utilities for signature verification system.
"""

import os
import pickle
import json
import numpy as np
import tensorflow as tf
import network
import preprocessor

class ModelLoader:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.custom_models = {}
        self.tf_sessions = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models."""
        if not os.path.exists(self.models_dir):
            print(f"Models directory {self.models_dir} not found!")
            return
        
        # Load custom neural network models
        for filename in os.listdir(self.models_dir):
            if filename.startswith('custom_nn_author_') and filename.endswith('.pkl'):
                author_id = filename.replace('custom_nn_author_', '').replace('.pkl', '')
                self.load_custom_model(author_id)
        
        print(f"Loaded {len(self.custom_models)} custom neural network models")
    
    def load_custom_model(self, author_id):
        """Load a custom neural network model for a specific author."""
        model_path = os.path.join(self.models_dir, f'custom_nn_author_{author_id}.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Recreate the network
            net = network.NeuralNetwork(model_data['sizes'])
            net.weights = model_data['weights']
            net.biases = model_data['biases']
            
            self.custom_models[author_id] = {
                'network': net,
                'trained_at': model_data.get('trained_at', 'Unknown')
            }
            
            print(f"Loaded custom NN model for author {author_id}")
            return True
        
        return False
    
    def load_tensorflow_model(self, author_id):
        """Load a TensorFlow model for a specific author."""
        model_dir = os.path.join(self.models_dir, f'tensorflow_author_{author_id}')
        model_path = os.path.join(model_dir, 'model.ckpt')
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        if os.path.exists(model_path + '.meta') and os.path.exists(metadata_path):
            # Read metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load the TensorFlow model
            tf.compat.v1.reset_default_graph()
            
            with tf.compat.v1.Session() as sess:
                # Restore the model
                saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
                saver.restore(sess, model_path)
                
                # Get the operations
                graph = tf.compat.v1.get_default_graph()
                x = graph.get_tensor_by_name("regression/input:0")
                y = graph.get_tensor_by_name("regression/output:0")
                
                self.tf_sessions[author_id] = {
                    'session': sess,
                    'input': x,
                    'output': y,
                    'metadata': metadata
                }
                
                print(f"Loaded TensorFlow model for author {author_id} (accuracy: {metadata['final_accuracy']:.4f})")
                return True
        
        return False
    
    def predict_custom_nn(self, image_data, author_id):
        """Make prediction using custom neural network."""
        if author_id not in self.custom_models:
            return None, "Model not found for author " + author_id
        
        try:
            # Preprocess the image
            processed_data = preprocessor.prepare(image_data)
            input_data = np.reshape(processed_data, (901, 1))
            
            # Get prediction
            net = self.custom_models[author_id]['network']
            output = net.feedforward(input_data)
            
            # Convert to probability (softmax-like)
            prob_forged = output[0][0]
            prob_genuine = output[1][0]
            
            # Normalize probabilities
            total = prob_forged + prob_genuine
            if total > 0:
                prob_forged /= total
                prob_genuine /= total
            
            prediction = "genuine" if prob_genuine > prob_forged else "forged"
            confidence = max(prob_genuine, prob_forged)
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'prob_genuine': float(prob_genuine),
                'prob_forged': float(prob_forged),
                'model_type': 'custom_nn',
                'author_id': author_id
            }, None
        
        except Exception as e:
            return None, f"Error in custom NN prediction: {str(e)}"
    
    def predict_tensorflow(self, image_data, author_id):
        """Make prediction using TensorFlow model."""
        # Note: TensorFlow session management needs to be handled differently
        # This is a simplified version - in practice, you'd want to keep sessions alive
        model_dir = os.path.join(self.models_dir, f'tensorflow_author_{author_id}')
        model_path = os.path.join(model_dir, 'model.ckpt')
        
        if not os.path.exists(model_path + '.meta'):
            return None, f"TensorFlow model not found for author {author_id}"
        
        try:
            # Preprocess the image
            processed_data = preprocessor.prepare(image_data)
            input_data = np.array([processed_data], dtype=np.float32)
            
            tf.compat.v1.reset_default_graph()
            
            with tf.compat.v1.Session() as sess:
                # Restore the model
                saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
                saver.restore(sess, model_path)
                
                # Get the operations
                graph = tf.compat.v1.get_default_graph()
                x = graph.get_tensor_by_name("regression/input:0")
                y = graph.get_tensor_by_name("regression/output:0")
                
                # Make prediction
                output = sess.run(y, feed_dict={x: input_data})
                
                prob_forged = output[0][0]
                prob_genuine = output[0][1]
                
                prediction = "genuine" if prob_genuine > prob_forged else "forged"
                confidence = max(prob_genuine, prob_forged)
                
                return {
                    'prediction': prediction,
                    'confidence': float(confidence),
                    'prob_genuine': float(prob_genuine),
                    'prob_forged': float(prob_forged),
                    'model_type': 'tensorflow',
                    'author_id': author_id
                }, None
        
        except Exception as e:
            return None, f"Error in TensorFlow prediction: {str(e)}"
    
    def get_available_authors(self):
        """Get list of available authors."""
        return list(self.custom_models.keys())
    
    def get_model_info(self):
        """Get information about loaded models."""
        info = {
            'custom_nn_models': len(self.custom_models),
            'available_authors': self.get_available_authors(),
            'model_details': {}
        }
        
        for author_id in self.custom_models:
            info['model_details'][author_id] = {
                'custom_nn_trained_at': self.custom_models[author_id]['trained_at']
            }
            
            # Check if TensorFlow model exists
            tf_metadata_path = os.path.join(self.models_dir, f'tensorflow_author_{author_id}', 'metadata.json')
            if os.path.exists(tf_metadata_path):
                with open(tf_metadata_path, 'r') as f:
                    tf_metadata = json.load(f)
                info['model_details'][author_id]['tensorflow_accuracy'] = tf_metadata['final_accuracy']
                info['model_details'][author_id]['tensorflow_trained_at'] = tf_metadata['trained_at']
        
        return info


# Global model loader instance
model_loader = None

def get_model_loader():
    """Get the global model loader instance."""
    global model_loader
    if model_loader is None:
        model_loader = ModelLoader()
    return model_loader
