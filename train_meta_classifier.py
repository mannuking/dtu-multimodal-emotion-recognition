import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")

# train_meta_classifier.py - Train meta-classifier combining PyTorch TER + TensorFlow FER/SER

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
from keras.optimizers import Adam
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
        # train_fer_tensorflow.py saves WEIGHTS only ({name}_best.weights.h5) —
        # the full .keras was never written by that trainer, so a .keras-only
        # lookup finds nothing and the meta-classifier silently gets no FER
        # models at all. Accept either artefact: a full .keras if present,
        # otherwise rebuild the architecture and load the weights into it.
        keras_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.keras")
        weights_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.weights.h5")

        if os.path.exists(keras_path):
            model = tf.keras.models.load_model(keras_path)
        elif os.path.exists(weights_path):
            from train_fer_tensorflow import build_vgg16_model, build_resnet50_model
            builder = build_resnet50_model if name.startswith("resnet50") else build_vgg16_model
            model = builder()
            model.load_weights(weights_path)
        else:
            print(f"❌ {name} not found (looked for {os.path.basename(keras_path)} "
                  f"and {os.path.basename(weights_path)})! Run train_fer_tensorflow.py first")
            return None
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

def resolve_speech_wav(p):
    """
    The triplet manifest stores speech_wav relative to the dataset root
    (e.g. 'angry/1022_ITS_ANG_XX.wav'), NOT relative to the repo root.
    Passing that straight to preprocess_audio() silently loads nothing and
    yields a zero vector for every row — the meta-classifier would then be
    trained on a constant feature and the fusion result would be garbage
    with no error anywhere.

    Try the path as given, then relative to combined_ser_dataset/, then
    relative to the repo root.
    """
    cands = [p,
             os.path.join(SER_COMBINED_DIR, p),
             os.path.join(SER_COMBINED_DIR, "combined_ser_dataset", p)]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def ser_predict_proba(model, wav_paths):
    """SER prediction"""
    arr = []
    missing = 0
    for p in wav_paths:
        real = resolve_speech_wav(p)
        if real is None:
            missing += 1
            arr.append(np.zeros(NUM_CLASSES))
            continue
        y = preprocess_audio(real)
        if y is None:
            missing += 1
            arr.append(np.zeros(NUM_CLASSES))
            continue
        arr.append(extract_features(y))
    if missing:
        print(f"  ⚠️  {missing}/{len(wav_paths)} audio files could not be resolved "
              f"— those rows get a zero vector")
    
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

    # ---- resolve the REAL visual channel ----
    # The triplets manifest is rebuilt by the pipeline's Stage 0b with
    # placeholder face_img values, so it can never carry the real frames.
    # fer_frames/manifest.csv is written by extract_fer_frames.py and is
    # joined here on the audio stem instead. The two jobs therefore never
    # write the same file and cannot corrupt each other.
    frame_index = {}
    fm = os.path.join("fer_frames", "manifest.csv")
    if os.path.exists(fm):
        fdf = pd.read_csv(fm)
        for sid, grp in fdf.groupby("sample_id"):
            paths = [str(x) for x in grp["frame_path"].tolist() if os.path.exists(str(x))]
            if paths:
                frame_index[str(sid)] = paths[len(paths) // 2]  # most expressive
        print(f"   face index: {len(frame_index)} clips with real frames")
    else:
        print("   ⚠️  no fer_frames/manifest.csv — the visual channel is empty")
        print("      Run scripts/extract_fer_frames.sbatch to populate it.")

    if frame_index:
        stems = df["speech_wav"].astype(str).map(lambda s: os.path.basename(s).rsplit(".", 1)[0])
        resolved = stems.map(frame_index)
        n_hit = int(resolved.notna().sum())
        df["face_img"] = resolved.fillna(df.get("face_img", pd.Series([""] * len(df))))
        print(f"   visual channel: {n_hit}/{len(df)} triplets resolved to a real face")
        if n_hit == 0:
            print("   ⚠️  0 resolved — every triplet will fuse a ZERO vector.")

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
    unresolvable = 0
    for wav_path in waves:
        if resolve_speech_wav(wav_path) is None:
            unresolvable += 1
        p_speech.append(ser_predict_proba(ser_model, [wav_path])[0])
    p_speech = np.array(p_speech)
    if unresolvable:
        print(f"  ❌ {unresolvable}/{len(waves)} triplet audio paths could not be "
              f"resolved against {SER_COMBINED_DIR}/")
        print("     Refusing to train the meta-classifier on zero vectors.")
        raise SystemExit(1)
    
    # Clear SER model
    del ser_model
    tf.keras.backend.clear_session()
    gc.collect()

    # Step 3: Load FER (TensorFlow) and get predictions
    print("\n3️⃣ FER Predictions (TensorFlow)...")
    fer_models = load_fer_tensorflow()
    p_face = []
    n_real_faces = 0
    n_missing_faces = 0
    batch_size = 8
    for i in range(0, len(imgs), batch_size):
        if i % 100 == 0:
            print(f"  Processing images {i}/{len(imgs)}...")
        
        batch_imgs = imgs[i:i+batch_size]
        batch = []
        for img_path in batch_imgs:
            try:
                if img_path and os.path.exists(img_path):
                    im = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                    arr = np.asarray(im).astype(np.float32)/255.0
                    n_real_faces += 1
                else:
                    arr = np.zeros((224,224,3), dtype=np.float32)
                    n_missing_faces += 1
            except Exception:
                arr = np.zeros((224,224,3), dtype=np.float32)
                n_missing_faces += 1
            batch.append(arr)
        
        batch = np.stack(batch, 0)
        batch_probs = fer_predict_proba(fer_models, batch)
        p_face.extend(batch_probs)
        
        del batch, batch_probs
        if i % (batch_size * 10) == 0:
            gc.collect()

    p_face = np.array(p_face)

    # Report the visual-channel coverage. If most rows are zeros the
    # "multimodal" fusion is really audio+text, and that has to be stated
    # rather than hidden behind a zero vector.
    total_faces = n_real_faces + n_missing_faces
    cov = (n_real_faces / total_faces * 100) if total_faces else 0.0
    print(f"\n   visual channel: {n_real_faces}/{total_faces} rows have a real "
          f"face ({cov:.1f}% coverage)")
    if cov < 50:
        print("   ⚠️  under 50% coverage — the fusion is mostly audio+text.")
        print("      Either run extract_fer_frames.py to fill the visual")
        print("      channel, or state the limitation explicitly in the paper.")

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
