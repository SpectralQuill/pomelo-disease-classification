import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

print("TensorFlow version:", tf.__version__)

try:
    model = EfficientNetB0(include_top=False,
                           input_shape=(224, 224, 3),
                           pooling="avg",
                           weights="imagenet")
    print("✅ EfficientNetB0 loaded successfully with 3-channel input.")
    print("Input shape actually used:", model.input_shape)
except Exception as e:
    print("❌ Still failing:")
    print(e)
