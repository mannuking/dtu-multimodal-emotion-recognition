# train_meta_classifier.py - Train meta-classifier combining PyTorch TER + TensorFlow FER/SER

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
from PIL import Image
import pickle

# PyTorch imports
import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

# TensorFlow imports
import tensorflow as tf
from keras import layers, Model
from keras.optimizers.legacy import Adam
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import librosa

from gpu_config import *
from gpu_runtime import enable_tf_perf, set_seed

# Perf runtime: mixed precision + XLA + multi-GPU MirroredStrategy
# Meta is a small MLP trained from scratch — mixed precision is safe.
STRATEGY = enable_tf_perf(mixed_precision=False)
set_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def map_emotion_to_unified(emotion_str, source=None):
    if emotion_str is None:
        return 6
    emotion_str = str(emotion_str).lower().strip()
    emotion_mapping = {
        'angry': 0, 'anger': 0, 'mad': 0, 'frustrated': 0,
        'disgust': 1, 'disgusted': 1,
        'fear': 2, 'scared': 2, 'afraid': 2,
        'happy': 3, 'happiness': 3, 'joy': 3, 'joyful': 3, 'excited': 3,
        'sad': 4, 'sadness': 4, 'sorrow': 4,
        'surprise': 5, 'surprised': 5,
        'neutral': 6, 'peaceful': 6, 'powerful': 6, 'calm': 6
    }
    if emotion_str in emotion_mapping:
        return emotion_mapping[emotion_str]
    try:
        emotion_num = int(emotion_str)
        if 0 <= emotion_num < NUM_CLASSES:
            return emotion_num
    except:
        pass
    return 6

# ==========================================
# LOAD PYTORCH TER MODEL
# ==========================================

@torch.no_grad()
def load_ter_pytorch():
    """Load trained PyTorch TER model"""
    ter_model_path = os.path.join(CHECKPOINT_DIR, "ter_pytorch_best.pt")
    ter_tokenizer_path = os.path.join(CHECKPOINT_DIR, "ter_pytorch_tokenizer")
    
    if not os.path.exists(ter_model_path):
        print("❌ TER model not found! Run train_ter_pytorch.py first")
        return None, None
    
    print("📥 Loading PyTorch TER model...")
    tokenizer = MobileBertTokenizer.from_pretrained(ter_tokenizer_path)
    model = MobileBertForSequenceClassification.from_pretrained(
        ter_tokenizer_path, num_labels=NUM_CLASSES
    ).to(device)
    model.load_state_dict(torch.load(ter_model_path, map_location=device))
    model.eval()
    
    print("✅ PyTorch TER model loaded")
    return model, tokenizer

@torch.no_grad()
def ter_predict_proba_pytorch(model, tokenizer, texts, maxlen=128):
    """PyTorch TER prediction"""
    if model is None:
        return np.ones((len(texts), NUM_CLASSES)) / NUM_CLASSES
    
    texts_cleaned = []
    for text in texts:
        if pd.isna(text) or text is None:
            text = "neutral"
        else:
            text = str(text).strip()
        texts_cleaned.append(text)
    
    tok = tokenizer(texts_cleaned, padding=True, truncation=True, max_length=maxlen, return_tensors='pt')
    tok = {k: v.to(device) for k, v in tok.items()}
    logits = model(**tok).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs

# ==========================================
# LOAD TENSORFLOW FER MODELS
# ==========================================

def load_fer_tensorflow():
    """Load trained TensorFlow FER models"""
    fer_models = []
    model_names = ["vgg16_orig", "vgg16_bal", "resnet50_orig", "resnet50_bal"]
    
    print("📥 Loading TensorFlow FER models...")
    for name in model_names:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.keras")
        if not os.path.exists(checkpoint_path):
            print(f"❌ {name} not found! Run train_fer_tensorflow.py first")
            return None
        model = tf.keras.models.load_model(checkpoint_path)
        fer_models.append(model)
        print(f"  ✅ Loaded {name}")
    
    return fer_models

def fer_predict_proba(models, img_array_batch):
    """Ensemble FER prediction"""
    preds = [m.predict(img_array_batch, verbose=0) for m in models]
    return np.mean(preds, axis=0)

# ==========================================
# LOAD TENSORFLOW SER MODEL
# ==========================================

def trim_silence(audio, thresh_scale=3):
    frame_length = 2048
    hop = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
    if len(rms) == 0:
        return audio
    thr = (rms.max() - rms.min()) / thresh_scale
    frames = np.nonzero(rms > thr)[0]
    if len(frames) == 0:
        return audio
    start = max(0, frames[0] * hop)
    end = min(len(audio), frames[-1] * hop)
    return audio[start:end]

def preprocess_audio(file_path):
    try:
        if os.path.exists(file_path):
            full_path = file_path
        elif file_path.startswith("combined_ser_dataset"):
            full_path = os.path.join(".", file_path)
        else:
            full_path = os.path.join(SER_COMBINED_DIR, file_path)
        
        if not os.path.exists(full_path):
            return None
        
        y, sr = librosa.load(full_path, sr=TARGET_SR)
        y = trim_silence(y)
        y = y[int(OFFSET_S*sr):]
        
        target = int(DUR_S * sr)
        if len(y) > target:
            y = y[:target]
        else:
            y = np.pad(y, (0, target - len(y)))
        
        return y
    except:
        return None

def extract_features(y, sr=TARGET_SR, n_mfcc=40):
    """
    Feature extraction matching the saved feature matrix layout
    (features-001.npy: 11,044 dims per sample = 251 frames × 44 features).

    Parameters: hop = 192 samples (12 ms), win = 400 samples (25 ms).
    This matches the training-time pipeline that produced ser_best.keras.
    """
    hop = 192
    win = 400
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win, hop_length=hop).T
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=win, hop_length=hop).T
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop).T
    energy = rms ** 2 * win
    prob = energy / (np.sum(energy) + 1e-8)
    entropy = -prob * np.log2(prob + 1e-12)
    feats = np.concatenate([zcr, rms, energy, entropy, mfcc], axis=1)
    return feats.flatten()


def load_ser_tensorflow():
    """Load trained TensorFlow SER model"""
    ser_checkpoint_path = os.path.join(CHECKPOINT_DIR, "ser_best.keras")
    
    if not os.path.exists(ser_checkpoint_path):
        print("❌ SER model not found! Run train_ser_tensorflow.py first")
        return None
    
    print("📥 Loading TensorFlow SER model...")
    # Load dummy to get input shape
    f = np.load(os.path.join(SER_FEATURES_DIR, "features.npy"))
    
    # Build model architecture (same as training)
    from keras import layers, Model
    I = layers.Input(shape=(f.shape[1], 1))
    x = layers.Conv1D(256, 5, padding='same')(I)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(256, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(512, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5, 2, 'same')(x)
    x = layers.Conv1D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(3, 2, 'same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(I, out)
    
    model.load_weights(ser_checkpoint_path)
    print("✅ TensorFlow SER model loaded")
    return model

def ser_predict_proba(model, wav_paths):
    """SER prediction"""
    arr = []
    for p in wav_paths:
        y = preprocess_audio(p)
        if y is None:
            arr.append(np.zeros(NUM_CLASSES))
            continue
        arr.append(extract_features(y))
    
    X = np.array(arr).reshape(len(arr), -1, 1)
    return model.predict(X, verbose=0)

# ==========================================
# META-CLASSIFIER
# ==========================================

def build_meta_classifier(input_dim=21):
    """Build meta-classifier (TensorFlow)"""
    I = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(I)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(I, out)

def train_meta_classifier():
    """Train hybrid meta-classifier"""
    meta_checkpoint_path = os.path.join(CHECKPOINT_DIR, "meta_hybrid_best.keras")

    if os.path.exists(meta_checkpoint_path):
        print("✅ Meta-classifier already trained!")
        return

    print("🔄 Training hybrid meta-classifier...")
    
    # Load triplet data
    df = pd.read_csv(TRIPLETS_MANIFEST)
    print(f"📊 Triplet data: {df.shape}")
    print(f"   columns: {list(df.columns)}")

    # Map each row to a synthetic text description (since the manifest has no
    # real text data). Same approach as train_ter_pytorch.py fallback.
    emotion_text = {
        'angry': 'I am feeling very angry right now',
        'disgust': 'This is completely disgusting',
        'fear': 'I am scared and afraid',
        'happy': 'I am so happy today',
        'sad': 'I feel very sad',
        'surprise': 'Wow what a surprise',
        'neutral': 'I am speaking normally',
    }
    df['text'] = df['label'].astype(str).str.lower().map(emotion_text).fillna('I am speaking')

    texts = df['text'].astype(str).tolist()
    waves = df['speech_wav'].tolist()
    imgs = df['face_img'].tolist()
    y_true = df['label'].apply(lambda x: map_emotion_to_unified(x)).values

    print("\n🔄 Getting predictions from all models...")
    
    # Step 1: Load TER (PyTorch) and get predictions
    print("\n1️⃣ TER Predictions (PyTorch)...")
    ter_model, ter_tokenizer = load_ter_pytorch()
    p_text = ter_predict_proba_pytorch(ter_model, ter_tokenizer, texts)
    
    # Clear PyTorch memory
    del ter_model, ter_tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    # Step 2: Load SER (TensorFlow) and get predictions
    print("\n2️⃣ SER Predictions (TensorFlow)...")
    ser_model = load_ser_tensorflow()
    p_speech = []
    for wav_path in waves:
        p_speech.append(ser_predict_proba(ser_model, [wav_path])[0])
    p_speech = np.array(p_speech)
    
    # Clear SER model
    del ser_model
    tf.keras.backend.clear_session()
    gc.collect()

    # Step 3: Load FER (TensorFlow) and get predictions
    print("\n3️⃣ FER Predictions (TensorFlow)...")
    fer_models = load_fer_tensorflow()
    p_face = []
    batch_size = 8
    for i in range(0, len(imgs), batch_size):
        if i % 100 == 0:
            print(f"  Processing images {i}/{len(imgs)}...")
        
        batch_imgs = imgs[i:i+batch_size]
        batch = []
        for img_path in batch_imgs:
            try:
                if os.path.exists(img_path):
                    im = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                    arr = np.asarray(im).astype(np.float32)/255.0
                else:
                    arr = np.zeros((224,224,3), dtype=np.float32)
            except:
                arr = np.zeros((224,224,3), dtype=np.float32)
            batch.append(arr)
        
        batch = np.stack(batch, 0)
        batch_probs = fer_predict_proba(fer_models, batch)
        p_face.extend(batch_probs)
        
        del batch, batch_probs
        if i % (batch_size * 10) == 0:
            gc.collect()

    p_face = np.array(p_face)
    
    # Clear FER models
    del fer_models
    tf.keras.backend.clear_session()
    gc.collect()

    # Combine features
    X = np.concatenate([p_text, p_speech, p_face], axis=1)
    print(f"\n📊 Combined features shape: X={X.shape}, y={y_true.shape}")

    # Split data
    Xtr, Xte, ytr, yte = train_test_split(X, y_true, test_size=0.2, stratify=y_true, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=SEED)

    # Build + compile inside strategy.scope() so MirroredStrategy can replicate it
    with STRATEGY.scope():
        # Meta is a small MLP trained from scratch — safe to use mixed precision.
        from gpu_runtime import enable_mixed_precision
        enable_mixed_precision()
        meta_model = build_meta_classifier(input_dim=X.shape[1])
        meta_model.compile(
            optimizer=Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=5,
            factor=0.5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            meta_checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]

    print("\n🏋️ Training meta-classifier...")
    
    history = meta_model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Test evaluation
    test_probs = meta_model.predict(Xte, verbose=0)
    test_pred = test_probs.argmax(1)
    test_acc = accuracy_score(yte, test_pred)
    
    print(f"\n✅ Meta-classifier Test Accuracy: {test_acc:.4f}")
    print(f"✅ Meta-classifier saved to: {meta_checkpoint_path}")

if __name__ == "__main__":
    train_meta_classifier()
