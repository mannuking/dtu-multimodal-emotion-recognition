# test_environment.py
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
except Exception as e:
    print(f"PyTorch error: {e}")

try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
except Exception as e:
    print(f"TensorFlow error: {e}")

try:
    import librosa
    print(f"Librosa: {librosa.__version__}")
except Exception as e:
    print(f"Librosa error: {e}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"OpenCV error: {e}")

try:
    import gradio as gr
    print(f"Gradio: {gr.__version__}")
except Exception as e:
    print(f"Gradio error: {e}")

try:
    from transformers import MobileBertTokenizer
    print("Transformers: OK")
except Exception as e:
    print(f"Transformers error: {e}")

print("\nEnvironment test completed!")
