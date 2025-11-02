from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
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
import cv2
from libs.pomelo_enhancer import rgba_to_bgr_avg_background, enhance_pomelo_image

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
CORS(app)

# Global variables
svm_model = None
feature_model = None
label_encoder = None
class_names = []
image_size = 224


@app.route("/")
def hello():
    return jsonify({"message": "Pomelooooooooooo!"})


def load_image_from_bytes(image_bytes, target_size):
    """Load image from bytes and preprocess for EfficientNet"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((target_size, target_size))
        img_array = kimage.img_to_array(img)
        return img_array
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        raise e


def load_models():
    """Load feature extractor (EfficientNetB0) and SVM classifier"""
    global svm_model, feature_model, label_encoder, class_names, image_size, scaler, selector

    try:
        print("🔧 Starting model loading process...")

        weights_dir = "weights"  # adjust if stored differently

        # === 1️⃣ Load SVM classifier ===
        svm_path = os.path.join(weights_dir, "svm_model.joblib")
        print(f"📦 Loading SVM model from {svm_path} ...")
        svm_model = joblib.load(svm_path)
        print("✓ SVM model loaded successfully")

        # === 2️⃣ Load label encoder ===
        le_path = os.path.join(weights_dir, "label_encoder.joblib")
        print(f"📄 Loading label encoder from {le_path} ...")
        label_encoder = joblib.load(le_path)
        class_names = label_encoder.classes_.tolist()
        print(f"✓ Loaded {len(class_names)} classes")

        # === 3️⃣ Load scaler ===
        scaler_path = os.path.join(weights_dir, "svm_scaler.joblib")
        print(f"⚖️ Loading scaler from {scaler_path} ...")
        scaler = joblib.load(scaler_path)
        print("✓ Scaler loaded successfully")

        # === 4️⃣ Load feature selector (if exists) ===
        selector_path = os.path.join(weights_dir, "svm_selector.joblib")
        selector = None
        if os.path.exists(selector_path):
            print(f"🔍 Loading feature selector from {selector_path} ...")
            selector = joblib.load(selector_path)
            print("✓ Feature selector loaded successfully")
        else:
            print("ℹ️ No feature selector found, using all features")

        # === 5️⃣ Load EfficientNet feature extractor ===
        eff_path = os.path.join(weights_dir, "final_model.keras")
        print(f"🧠 Loading EfficientNetB0 feature extractor from {eff_path} ...")
        base_model = tf.keras.models.load_model(eff_path)
        print("✓ Base model loaded")

        # Derive feature extractor from GAP layer (matches training)
        print("🔍 Building feature extractor from GAP layer ...")
        try:
            feature_layer = base_model.get_layer("gap").output
        except ValueError:
            print("⚠️ GAP layer not found by name, using last layer instead.")
            feature_layer = base_model.layers[-2].output

        feature_model = tf.keras.Model(inputs=base_model.input, outputs=feature_layer)

        # Determine input size from model config
        image_size = base_model.input_shape[1]
        print(f"📐 Input image size: {image_size}px")
        print(f"🎉 Models loaded successfully with {len(class_names)} classes")

    except Exception as e:
        print(f"💥 Error loading models: {e}")
        traceback.print_exc()
        raise e
    

@app.route("/health", methods=["GET"])
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


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        image_bytes = image_file.read()
        print(f"📨 Received image: {len(image_bytes)} bytes")

        # --- Load as raw image (including alpha if present) ---
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image_bgra = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if image_bgra is None:
            return jsonify({"error": "Unable to decode image"}), 400

        # --- Convert RGBA → BGR or ensure valid BGR ---
        if image_bgra.shape[-1] == 4:
            image_bgr = rgba_to_bgr_avg_background(image_bgra)
        elif image_bgra.shape[-1] == 3:
            image_bgr = image_bgra
        else:
            return jsonify({"error": f"Unsupported image format {image_bgra.shape}"}), 400

        # --- Apply Pomelo enhancement (CLAHE, color rebalance, etc.) ---
        print("🌿 Applying Pomelo enhancement...")
        enhanced_bgr = enhance_pomelo_image(image_bgr)

        # --- Convert to RGB and resize for EfficientNet ---
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        enhanced_rgb = cv2.resize(enhanced_rgb, (image_size, image_size))
        img_array = np.expand_dims(enhanced_rgb, axis=0).astype(np.float32)
        img_array = preprocess_input(img_array)

        # --- Extract features ---
        print("🔍 Extracting features...")
        features = feature_model.predict(img_array, verbose=0)
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)

        print(f"📊 Raw feature shape: {features.shape}")

        # --- Apply same preprocessing as training ---
        print("⚖️ Applying scaling...")
        features_scaled = scaler.transform(features)
        
        if selector is not None:
            print(f"🔍 Applying feature selection ({features_scaled.shape[1]} -> {selector.transform(features_scaled).shape[1]} features)")
            features_final = selector.transform(features_scaled)
        else:
            features_final = features_scaled

        print(f"📊 Final feature shape: {features_final.shape}")

        # --- SVM prediction ---
        print("🤖 Running SVM classifier...")
        if hasattr(svm_model, "predict_proba"):
            probs = svm_model.predict_proba(features_final)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
        else:
            pred_idx = int(svm_model.predict(features_final)[0])
            confidence = 1.0

        pred_class = class_names[pred_idx]

        # --- Build class probability map ---
        if hasattr(svm_model, "predict_proba"):
            class_probs = {class_names[i]: float(prob) for i, prob in enumerate(probs)}
        else:
            class_probs = {c: 0.0 for c in class_names}
            class_probs[pred_class] = confidence

        print(f"🎯 Prediction: {pred_class} (conf: {confidence:.3f})")

        return jsonify({
            "predicted_class": pred_class,
            "confidence": confidence,
            "all_predictions": class_probs,
            "status": "success"
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/test", methods=["GET"])
def test_endpoint():
    return jsonify({"message": "Server is running!", "status": "success"})


if __name__ == "__main__":
    try:
        print("=" * 50)
        print("🚀 Starting Pomelo Disease Classification Backend...")
        print("=" * 50)

        # Load .env (optional)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        dotenv_path = os.path.join(base_dir, ".env")
        load_dotenv(dotenv_path)

        host = os.getenv("FLASK_HOST", "0.0.0.0")
        port = int(os.getenv("FLASK_PORT", 5000))

        # Load models
        load_models()

        print(f"🌐 Server ready at http://{host}:{port}")
        app.run(host=host, port=port, debug=False)

    except Exception as e:
        print("💥 CRITICAL ERROR - Server failed to start!")
        traceback.print_exc()
        sys.exit(1)
