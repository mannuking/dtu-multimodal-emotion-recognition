"""
predict_emotion.py — Run the trained wav2vec2 SER on any audio file.

Usage:
    python predict_emotion.py path/to/audio.wav

Outputs the predicted emotion label + probability distribution across
all 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise).

Also batch mode:
    python predict_emotion.py path/to/folder/
"""
import os, sys, pickle, warnings
import numpy as np
import librosa
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")

CHECKPOINT_DIR = "model_checkpoints"
TARGET_SR = 16000
MAX_S = 6.0


def load_audio(path, max_seconds=MAX_S, sr=TARGET_SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    max_samples = int(max_seconds * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    elif len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)), mode="constant")
    if np.abs(y).max() > 0:
        y = y / np.abs(y).max()
    return y.astype(np.float32)


def predict(model, x, device):
    x_t = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_emotion.py <audio_file_or_folder>")
        sys.exit(1)

    input_path = sys.argv[1]

    ser_ckpt = os.path.join(CHECKPOINT_DIR, "ser_best.pt")
    ser_encoder = os.path.join(CHECKPOINT_DIR, "ser_label_encoder.pkl")
    if not os.path.exists(ser_ckpt) or not os.path.exists(ser_encoder):
        print(f"ERROR: checkpoints not found in {CHECKPOINT_DIR}/")
        sys.exit(1)

    with open(ser_encoder, "rb") as f:
        label_encoder = pickle.load(f)
    classes = label_encoder.classes_
    print(f"Loaded model with {len(classes)} classes: {list(classes)}")

    # Load model
    sys.path.insert(0, ".")
    from train_ser_wav2vec import Wav2Vec2SER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2SER(num_classes=len(classes)).to(device)
    state = torch.load(ser_ckpt, map_location=device, weights_only=False)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()

    # Collect audio files
    if os.path.isdir(input_path):
        paths = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith((".wav", ".mp3", ".flac"))
        ])
    else:
        paths = [input_path]

    print(f"\nPredicting {len(paths)} audio file(s):\n")
    for path in paths:
        try:
            x = load_audio(path)
            probs = predict(model, x, device)
            pred_idx = int(np.argmax(probs))
            pred_label = classes[pred_idx]
            print(f"\u2709\ufe0f  {os.path.basename(path)}")
            print(f"   predicted: {pred_label} (confidence {probs[pred_idx]:.2%})")
            print(f"   probabilities:")
            for cls, p in zip(classes, probs):
                bar = "\u2588" * int(p * 40)
                marker = " <--" if cls == pred_label else ""
                print(f"     {cls:9s}: {p:.3f}  {bar}{marker}")
            print()
        except Exception as e:
            print(f"   \u274c {path}: {e}")


if __name__ == "__main__":
    main()