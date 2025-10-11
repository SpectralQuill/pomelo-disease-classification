from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as kimage
import joblib
import io
import os
from PIL import Image
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import traceback
import sys

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
CORS(app)

# Global variables for loaded models
svm_model = None
feature_model = None
class_names = []
image_size = 224

@app.route('/')
def hello():
    return jsonify({"message": "Pomelooooooooooo!"})

def load_image_from_bytes(image_bytes, target_size):
    """Load image from bytes and preprocess for EfficientNet"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((target_size, target_size))
        img_array = kimage.img_to_array(img)
        return img_array
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        raise e

def load_models():
    """Load the SVM and feature extraction models"""
    global svm_model, feature_model, class_names, image_size
    
    try:
        print("🔧 Starting model loading process...")
        
        # Load SVM model
        print("📦 Loading SVM model...")
        svm_model = joblib.load('weights/svm_classifier.joblib')
        print("✓ SVM model loaded")
        
        # Load model info
        print("📄 Loading model info...")
        import json
        with open('weights/model_info.json', 'r') as f:
            model_info = json.load(f)
        
        class_names = model_info['class_names']
        image_size = model_info['config'].get('image_size', 224)
        print(f"✓ Loaded class names: {class_names}")
        
        # Load the complete fine-tuned model
        print("🧠 Loading EfficientNet model...")
        complete_model = keras.models.load_model('weights/efficientnetb0_finetuned.keras')
        print("✓ EfficientNet model loaded")
        
        # SIMPLIFIED APPROACH: Use a known working layer
        print("🔍 Setting up feature extraction...")
        
        # Test different output layers to find one that works
        test_input = np.random.random((1, image_size, image_size, 3)).astype(np.float32)
        
        # Try common layer indices that often work
        layer_indices_to_try = [
            -2,  # Second to last layer (usually before final classification)
            -3,  # Third to last layer
            1,   # Usually the base model
        ]
        
        # Also try to find layers by name
        layer_names_to_try = [
            'efficientnetb0',
            'global_average_pooling2d',
            'avg_pool',
            'top_activation',
        ]
        
        feature_model = None
        
        # First try by name
        for layer_name in layer_names_to_try:
            for layer in complete_model.layers:
                if layer_name in layer.name.lower():
                    try:
                        print(f"🔬 Testing layer: {layer.name}")
                        test_model = keras.models.Model(
                            inputs=complete_model.input, 
                            outputs=layer.output
                        )
                        test_output = test_model.predict(test_input, verbose=0)
                        # Flatten if needed
                        if len(test_output.shape) > 2:
                            test_output = test_output.reshape(test_output.shape[0], -1)
                        print(f"   ✅ Layer {layer.name} works! Output shape: {test_output.shape}")
                        feature_model = test_model
                        break
                    except Exception as e:
                        print(f"   ❌ Layer {layer.name} failed: {e}")
                        continue
            if feature_model:
                break
        
        # If no layer found by name, try by index
        if not feature_model:
            for layer_idx in layer_indices_to_try:
                try:
                    layer = complete_model.layers[layer_idx]
                    print(f"🔬 Testing layer {layer_idx}: {layer.name}")
                    test_model = keras.models.Model(
                        inputs=complete_model.input, 
                        outputs=layer.output
                    )
                    test_output = test_model.predict(test_input, verbose=0)
                    # Flatten if needed
                    if len(test_output.shape) > 2:
                        test_output = test_output.reshape(test_output.shape[0], -1)
                    print(f"   ✅ Layer {layer.name} works! Output shape: {test_output.shape}")
                    feature_model = test_model
                    break
                except Exception as e:
                    print(f"   ❌ Layer {layer_idx} failed: {e}")
                    continue
        
        # Final fallback: use the complete model
        if not feature_model:
            print("⚠️ Using complete model as fallback")
            feature_model = complete_model
        
        # Test the final feature extraction
        print("🧪 Final feature extraction test...")
        test_features = feature_model.predict(test_input, verbose=0)
        if len(test_features.shape) > 2:
            test_features = test_features.reshape(test_features.shape[0], -1)
        print(f"✓ Final feature shape: {test_features.shape}")
        print(f"✓ Feature dimension: {test_features.shape[-1]}")
        
        print("🎉 All models loaded successfully!")
        print(f"📊 Ready for classification with {len(class_names)} classes")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            "status": "healthy", 
            "models_loaded": svm_model is not None and feature_model is not None,
            "class_names": class_names,
            "image_size": image_size
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
            
        image_bytes = image_file.read()
        print(f"📨 Received image: {len(image_bytes)} bytes")
        
        # Load and preprocess image
        img_array = load_image_from_bytes(image_bytes, image_size)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        print("🔍 Extracting features...")
        features = feature_model.predict(img_array, verbose=0)
        
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        print(f"📊 Features extracted: {features.shape}")
        
        # SVM prediction
        print("🤖 Running SVM classification...")
        if hasattr(svm_model, "predict_proba"):
            probabilities = svm_model.predict_proba(features)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_idx])
        else:
            predicted_class_idx = int(svm_model.predict(features)[0])
            confidence = 1.0
        
        predicted_class = class_names[predicted_class_idx]
        
        # Prepare response with all class probabilities
        if hasattr(svm_model, "predict_proba"):
            class_probs = {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        else:
            class_probs = {class_names[i]: 0.0 for i in range(len(class_names))}
            class_probs[predicted_class] = confidence
        
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": class_probs,
            "status": "success"
        }
        
        print(f"🎯 Prediction: {predicted_class} (confidence: {confidence:.3f})")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        print("🔍 Prediction traceback:")
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to verify server is running"""
    return jsonify({"message": "Server is running!", "status": "success"})

if __name__ == '__main__':
    try:
        print("=" * 50)
        print("🚀 Starting Pomelo Disease Classification Backend...")
        print("=" * 50)

        # --- Load .env from project root (two levels up from this file) ---
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        dotenv_path = os.path.join(base_dir, ".env")
        load_dotenv(dotenv_path)

        # --- Read environment variables ---
        host = os.getenv("FLASK_HOST", "0.0.0.0")
        port = int(os.getenv("FLASK_PORT", 5000))

        # --- Load models ---
        load_models()

        print(f"🌐 Server starting on http://{host}:{port}")
        print("💡 Test with: curl http://localhost:5000/health")
        print("💡 Test prediction with: curl -X POST -F 'image=@test.jpg' http://localhost:5000/predict")

        # --- Start Flask server ---
        app.run(host=host, port=port, debug=False)

    except Exception as e:
        print("💥 CRITICAL ERROR - Server failed to start!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("❌ Exiting...")
        sys.exit(1)
