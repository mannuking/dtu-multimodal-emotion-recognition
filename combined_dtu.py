# Core imports and setup
import os, sys, math, json, random, shutil, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import TFMobileBertForSequenceClassification, MobileBertTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Constants
SEED = 42
NUM_CLASSES = 7
EMOTION_ORDER = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# Data paths
FER_DATA_DIR = "fer_dataset"
FER_TRAIN_DIR = os.path.join(FER_DATA_DIR, "train")
FER_TEST_DIR = os.path.join(FER_DATA_DIR, "test")
SER_COMBINED_DIR = "combined_ser_dataset"
SER_METADATA_CSV = os.path.join(SER_COMBINED_DIR, "metadata.csv")
SER_FEATURES_DIR = "ser_feature_output"
TEXT_TRAIN_CSV = "ter_dataset/merged/text_train.csv"
TEXT_VAL_CSV = "ter_dataset/merged/text_val.csv"
TEXT_TEST_CSV = "ter_dataset/merged/text_test.csv"
TRIPLETS_MANIFEST = "triplets_manifest_balanced.csv"

# Runtime test files
RUNTIME_CAPTURE_TEXT = "What a great day! I'm so excited to learn."
RUNTIME_CAPTURE_WAV = "test_audio.wav"
RUNTIME_CAPTURE_IMAGE = "picture.png"

# Model checkpoints directory
CHECKPOINT_DIR = "model_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Configuration
USE_PRETRAINED_FOR_FACE = False
USE_PRETRAINED_TEXT = True

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Emotion mappings
EMOTION_TO_ID = {e:i for i,e in enumerate(EMOTION_ORDER)}
ID_TO_EMOTION = {i:e for e,i in EMOTION_TO_ID.items()}

print(f"Using TensorFlow version: {tf.__version__}")

# ======================= UTILITY FUNCTIONS =======================
def to_categorical(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    for i,yi in enumerate(y):
        out[i, yi] = 1.0
    return out

def save_checkpoint(checkpoint_name, data):
    """Save checkpoint to drive"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.json")
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_name):
    """Load checkpoint from drive"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        return data
    return None

def checkpoint_exists(checkpoint_name):
    """Check if checkpoint exists"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.json")
    return os.path.exists(checkpoint_path)

# ======================= FER (FACIAL EMOTION RECOGNITION) =======================
IMG_SIZE = (224,224)
FER_BATCH = 32
FER_EPOCHS = 30

def build_vgg16_model(input_shape=(224,224,3), num_classes=NUM_CLASSES, pretrained=USE_PRETRAINED_FOR_FACE):
    base = tf.keras.applications.VGG16(
        include_top=False, weights='imagenet' if pretrained else None, input_shape=input_shape
    )
    
    if pretrained:
        for l in base.layers[:-8]:
            l.trainable=False
    
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(base.input, out)

def build_resnet50_model(input_shape=(224,224,3), num_classes=NUM_CLASSES, pretrained=USE_PRETRAINED_FOR_FACE):
    base = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet' if pretrained else None, input_shape=input_shape
    )
    
    if pretrained:
        for l in base.layers[:-10]:
            l.trainable=False
    
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(base.input, out)

def fer_data_generators():
    aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    noaug = ImageDataGenerator(rescale=1./255)
    testgen = ImageDataGenerator(rescale=1./255)
    
    train_orig = noaug.flow_from_directory(
        FER_TRAIN_DIR, target_size=IMG_SIZE, color_mode='rgb',
        batch_size=FER_BATCH, class_mode='categorical', shuffle=True, seed=SEED
    )
    
    train_aug = aug.flow_from_directory(
        FER_TRAIN_DIR, target_size=IMG_SIZE, color_mode='rgb',
        batch_size=FER_BATCH, class_mode='categorical', shuffle=True, seed=SEED
    )
    
    val = testgen.flow_from_directory(
        FER_TEST_DIR, target_size=IMG_SIZE, color_mode='rgb',
        batch_size=FER_BATCH, class_mode='categorical', shuffle=False
    )
    
    test = testgen.flow_from_directory(
        FER_TEST_DIR, target_size=IMG_SIZE, color_mode='rgb',
        batch_size=FER_BATCH, class_mode='categorical', shuffle=False
    )
    
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_orig.classes),
        y=train_orig.classes
    )
    weights = dict(enumerate(weights))
    return {"orig":train_orig, "balanced":train_aug, "val":val, "test":test, "weights":weights}

def train_fer_model(model, name, train_gen, val_gen, class_weights):
    model_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.keras")
    
    model.compile(
        optimizer=Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.4,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            model_checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples//train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples//val_gen.batch_size,
        epochs=FER_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Load best weights
    model.load_weights(model_checkpoint_path)
    
    # Save training history
    history_checkpoint = {
        "history": {
            "loss": [float(x) for x in history.history.get('loss', [])],
            "accuracy": [float(x) for x in history.history.get('accuracy', [])],
            "val_loss": [float(x) for x in history.history.get('val_loss', [])],
            "val_accuracy": [float(x) for x in history.history.get('val_accuracy', [])]
        }
    }
    save_checkpoint(f"{name}_history", history_checkpoint)
    return model

def fer_predict_proba(models, img_array_batch):
    preds = [m.predict(img_array_batch, verbose=0) for m in models]
    return np.mean(preds, axis=0)

def train_fer_models_with_checkpoints():
    """Train FER models with checkpoint recovery"""
    print("🚀 Starting FER model training...")
    
    # Check for existing checkpoints
    checkpoint_status = load_checkpoint("fer_training_status")
    completed_models = set(checkpoint_status.get("completed_models", [])) if checkpoint_status else set()
    
    # Load data generators
    print("Loading data generators...")
    gens = fer_data_generators()
    fer_models = {}
    
    model_configs = [
        ("vgg16_orig", lambda: build_vgg16_model(), "orig"),
        ("vgg16_bal", lambda: build_vgg16_model(), "balanced"),
        ("resnet50_orig", lambda: build_resnet50_model(), "orig"),
        ("resnet50_bal", lambda: build_resnet50_model(), "balanced")
    ]
    
    for model_name, model_builder, gen_type in model_configs:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.keras")
        
        if model_name in completed_models and os.path.exists(checkpoint_path):
            print(f"✅ Loading existing {model_name} from checkpoint...")
            model = model_builder()
            model.load_weights(checkpoint_path)
            fer_models[model_name] = model
        else:
            print(f"🔄 Training {model_name}...")
            model = model_builder()
            trained_model = train_fer_model(
                model, model_name, gens[gen_type], gens['val'], gens['weights']
            )
            fer_models[model_name] = trained_model
            
            # Update checkpoint status
            completed_models.add(model_name)
            save_checkpoint("fer_training_status", {"completed_models": list(completed_models)})
    
    # Create ensemble list
    fer_ensemble = [
        fer_models["vgg16_orig"],
        fer_models["vgg16_bal"],
        fer_models["resnet50_orig"],
        fer_models["resnet50_bal"]
    ]
    
    print("✅ FER training complete!")
    return fer_ensemble

# ======================= SER (SPEECH EMOTION RECOGNITION) =======================

# Audio processing constants
TARGET_SR = 16000
OFFSET_S = 0.6
DUR_S = 2.5

def trim_silence(audio, thresh_scale=3):
    frame_length=2048
    hop=512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
    if len(rms)==0:
        return audio
    thr=(rms.max()+rms.min())/thresh_scale
    frames = np.nonzero(rms>thr)[0]
    if len(frames)==0:
        return audio
    start=max(0, frames[0]*hop)
    end=min(len(audio), frames[-1]*hop)
    return audio[start:end]

def preprocess_audio(file_path):
    """Preprocess audio with smart path resolution"""
    try:
        # Smart path resolution
        if os.path.exists(file_path):
            full_path = file_path
        elif file_path.startswith('combined_ser_dataset'):
            # Path is relative to base DTU directory
            full_path = os.path.join('./', file_path)
        else:
            # Try relative to SER_COMBINED_DIR
            full_path = os.path.join(SER_COMBINED_DIR, file_path)
        
        if not os.path.exists(full_path):
            print(f"⚠️ File not found: {file_path} (tried: {full_path})")
            return None
            
        y, sr = librosa.load(full_path, sr=TARGET_SR)
        y = trim_silence(y)
        y = y[int(OFFSET_S*sr):]
        target = int(DUR_S*sr)
        if len(y) > target:
            y = y[:target]
        else:
            y = np.pad(y, (0, target-len(y)))
        return y
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def augment_audio(y):
    out=[]
    # Add noise
    out.append(y + 0.005*np.random.normal(size=len(y)))
    # Pitch shift
    out.append(librosa.effects.pitch_shift(y=y, sr=TARGET_SR, n_steps=random.choice([-2,-1,1,2])))
    # Time stretch
    rate=random.uniform(0.9,1.1)
    s=librosa.effects.time_stretch(y=y, rate=rate)
    s = s[:len(y)] if len(s)>len(y) else np.pad(s,(0,len(y)-len(s)))
    out.append(s)
    # Time shift
    out.append(np.roll(y, int(random.uniform(-0.1,0.1)*len(y))))
    return out

def extract_features(y, sr=TARGET_SR, n_mfcc=40):
    hop=int(0.010*sr)
    win=int(0.025*sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win, hop_length=hop).T
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=win, hop_length=hop).T
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop).T
    energy = (rms**2)*win
    prob = energy/(np.sum(energy)+1e-8)
    entropy = -(prob*np.log2(prob+1e-12))
    feats = np.concatenate([zcr, rms, energy, entropy, mfcc], axis=1)
    return feats.flatten()

def generate_ser_embeddings():
    """Generate and save SER features with checkpoint recovery"""
    features_path = os.path.join(SER_FEATURES_DIR, "features.npy")
    labels_path = os.path.join(SER_FEATURES_DIR, "labels.npy")
    
    # Check if features already exist
    if os.path.exists(features_path) and os.path.exists(labels_path):
        print("✅ SER features already exist, skipping generation...")
        return
    
    os.makedirs(SER_FEATURES_DIR, exist_ok=True)
    print("🔄 Generating SER features...")
    
    meta = pd.read_csv(SER_METADATA_CSV)
    feats, labels = [], []
    
    for i, (_, row) in enumerate(meta.iterrows()):
        if i % 100 == 0:
            print(f"Processing audio {i+1}/{len(meta)}")
        
        wav = os.path.join(SER_COMBINED_DIR, row['filepath'])
        y = preprocess_audio(wav)
        if y is None:
            continue
        
        # Original features
        feats.append(extract_features(y))
        labels.append(row['emotion'])
        
        # Augmented features
        for aug in augment_audio(y):
            feats.append(extract_features(aug))
            labels.append(row['emotion'])
    
    # Save features
    np.save(features_path, np.array(feats))
    np.save(labels_path, np.array(labels))
    print(f"✅ SER features saved: {len(feats)} samples")

def load_ser_embeddings():
    f = np.load(os.path.join(SER_FEATURES_DIR,"features.npy"))
    l = np.load(os.path.join(SER_FEATURES_DIR,"labels.npy"))
    
    # Normalize labels (lowercase, strip spaces)
    l = np.array([str(lbl).strip().lower() for lbl in l])
    
    # Define consistent mapping (lowercase)
    EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    label_map = {e:i for i,e in enumerate(EMOTION_ORDER)}
    
    # Filter out invalid labels
    mask = np.array([lbl in label_map for lbl in l])
    f, l = f[mask], l[mask]
    
    # Map to integers
    y = np.array([label_map[lbl] for lbl in l])
    X = f.reshape(f.shape[0], f.shape[1], 1)
    return X, y, label_map

def build_ser_1dcnn(input_shape, num_classes=NUM_CLASSES):
    I = layers.Input(shape=input_shape)
    
    # Conv blocks
    x = layers.Conv1D(256,5,padding='same')(I)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5,2,'same')(x)
    
    x = layers.Conv1D(256,5,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5,2,'same')(x)
    
    x = layers.Conv1D(512,5,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5,2,'same')(x)
    
    x = layers.Conv1D(512,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5,2,'same')(x)
    
    x = layers.Conv1D(256,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5,2,'same')(x)
    
    x = layers.Conv1D(256,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5,2,'same')(x)
    
    x = layers.Conv1D(128,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5,2,'same')(x)
    
    x = layers.Conv1D(128,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(5,2,'same')(x)
    
    x = layers.Conv1D(64,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(3,2,'same')(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(I, out)

def train_ser_model():
    """Train SER model with checkpoint recovery"""
    ser_checkpoint_path = os.path.join(CHECKPOINT_DIR, "ser_best.keras")
    ser_encoder_path = os.path.join(CHECKPOINT_DIR, "ser_label_encoder.pkl")
    
    # Check for existing checkpoint
    if os.path.exists(ser_checkpoint_path) and os.path.exists(ser_encoder_path):
        print("✅ Loading existing SER model from checkpoint...")
        X, y, le = load_ser_embeddings()
        model = build_ser_1dcnn((X.shape[1],1))
        model.load_weights(ser_checkpoint_path)
        
        # Load label encoder
        with open(ser_encoder_path, 'rb') as f:
            le = pickle.load(f)
        return model, le
    
    print("🔄 Training SER model...")
    
    # Generate features if needed
    if not (os.path.exists(os.path.join(SER_FEATURES_DIR,"features.npy")) and 
            os.path.exists(os.path.join(SER_FEATURES_DIR,"labels.npy"))):
        generate_ser_embeddings()
    
    # Load data
    X, y, le = load_ser_embeddings()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.111, stratify=ytr, random_state=SEED)
    
    # Build and compile model
    model = build_ser_1dcnn((X.shape[1],1))
    model.compile(
        optimizer=Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks with checkpoint saving
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=3,
            factor=0.1,
            min_lr=1e-5
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            ser_checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva,yva),
        epochs=1,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best weights
    model.load_weights(ser_checkpoint_path)
    
    # Save label encoder
    with open(ser_encoder_path, 'wb') as f:
        pickle.dump(le, f)
    
    # Test evaluation
    te_probs = model.predict(Xte, verbose=0)
    te_pred = te_probs.argmax(1)
    test_acc = (te_pred==yte).mean()
    print(f"SER Test Accuracy: {test_acc:.4f}")
    
    # Save training history and results
    history_checkpoint = {
        "history": {
            "loss": [float(x) for x in history.history.get('loss', [])],
            "accuracy": [float(x) for x in history.history.get('accuracy', [])],
            "val_loss": [float(x) for x in history.history.get('val_loss', [])],
            "val_accuracy": [float(x) for x in history.history.get('val_accuracy', [])]
        },
        "test_accuracy": float(test_acc)
    }
    save_checkpoint("ser_training_results", history_checkpoint)
    
    return model, le

def ser_predict_proba(model, wav_paths):
    arr=[]
    for p in wav_paths:
        y = preprocess_audio(p)
        if y is None:
            arr.append(np.zeros(NUM_CLASSES))
            continue
        arr.append(extract_features(y))
    X = np.array(arr).reshape(len(arr), -1, 1)
    return model.predict(X, verbose=0)

# ======================= TER (TEXT EMOTION RECOGNITION) =======================

def map_emotion_to_unified(emotion_str, source=None):
    """Map emotion from any dataset to unified 7-class system"""
    if emotion_str is None:
        return 6  # neutral
    
    emotion_str = str(emotion_str).lower().strip()
    
    # Unified emotion mapping
    emotion_mapping = {
        # anger (0)
        'angry': 0, 'anger': 0, 'mad': 0, 'frustrated': 0,
        # disgust (1)
        'disgust': 1, 'disgusted': 1,
        # fear (2)
        'fear': 2, 'scared': 2, 'afraid': 2,
        # happy (3)
        'happy': 3, 'happiness': 3, 'joy': 3, 'joyful': 3, 'excited': 3,
        # sad (4)
        'sad': 4, 'sadness': 4, 'sorrow': 4,
        # surprise (5)
        'surprise': 5, 'surprised': 5,
        # neutral (6)
        'neutral': 6, 'peaceful': 6, 'powerful': 6, 'calm': 6
    }
    
    # Try unified mapping first
    if emotion_str in emotion_mapping:
        return emotion_mapping[emotion_str]
    
    # Try numeric conversion
    try:
        emotion_num = int(emotion_str)
        if 0 <= emotion_num < NUM_CLASSES:
            return emotion_num
    except:
        pass
    
    # Default to neutral
    return 6

def load_text_csv(path):
    """Load text CSV with robust parsing for combined datasets"""
    try:
        # Try different delimiters and parsing strategies
        delimiters = [';', ',', '\t']
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        df = None
        
        for delimiter in delimiters:
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, delimiter=delimiter, encoding=encoding,
                                   quotechar='"', on_bad_lines='skip')
                    if df.shape[1] >= 2:  # At least 2 columns
                        print(f"✅ Successfully loaded with delimiter '{delimiter}' and encoding '{encoding}'")
                        break
                except:
                    continue
            else:
                continue
            break
        
        if df is None:
            print("⚠️ Standard parsing failed, trying manual parsing...")
            return load_text_csv_manual(path)
        
        # Auto-detect columns
        text_col = None
        emotion_col = None
        dataset_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['utterance', 'text', 'sentence', 'phrase', 'dialogue']):
                text_col = col
            if any(keyword in col_lower for keyword in ['emotion', 'mapped', 'label', 'sentiment', 'class']):
                emotion_col = col
            if any(keyword in col_lower for keyword in ['dataset', 'source', 'corpus']):
                dataset_col = col
        
        # Fallback to first two columns if auto-detection fails
        if text_col is None and emotion_col is None and len(df.columns) >= 2:
            text_col = df.columns[0]
            emotion_col = df.columns[1]
            print(f"⚠️ Using first two columns: text='{text_col}', emotion='{emotion_col}'")
        
        if text_col is None or emotion_col is None:
            raise ValueError(f"Could not identify text and emotion columns in {path}")
        
        # Extract and clean data
        texts = df[text_col].astype(str).str.strip().values
        emotions = df[emotion_col].values
        
        # Map emotions to unified system
        print("🔄 Mapping emotions to unified system...")
        mapped_emotions = []
        for i, emotion in enumerate(emotions):
            source = df[dataset_col].iloc[i] if dataset_col is not None and dataset_col in df.columns else None
            mapped_emotions.append(map_emotion_to_unified(emotion, source))
        emotions = np.array(mapped_emotions)
        
        # Validate labels
        if np.any(emotions < 0) or np.any(emotions >= NUM_CLASSES):
            print(f"⚠️ Invalid labels found: {emotions.min()} to {emotions.max()} (should be 0-{NUM_CLASSES-1})")
            invalid_mask = (emotions < 0) | (emotions >= NUM_CLASSES)
            emotions[invalid_mask] = 6  # Map to Neutral
            print(f"Fixed {invalid_mask.sum()} invalid labels")
        
        # Print class distribution
        unique, counts = np.unique(emotions, return_counts=True)
        emotion_names = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        print("📊 Class distribution:")
        for cls, count in zip(unique, counts):
            print(f"  {emotion_names[cls] if cls < len(emotion_names) else cls}: {count} samples ({count/len(emotions)*100:.1f}%)")
        
        print(f"✅ Loaded {len(texts)} samples")
        return texts, emotions
    
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        print("🔄 Trying manual parsing...")
        return load_text_csv_manual(path)

def load_text_csv_manual(path):
    """Manual CSV parsing for problematic files"""
    texts = []
    emotions = []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Detect header and delimiter
        first_line = lines[0].strip()
        has_header = any(keyword in first_line.lower() for keyword in ['utterance', 'emotion', 'text', 'label'])
        delimiter = ';' if ';' in first_line else ','
        start_idx = 1 if has_header else 0
        
        for i, line in enumerate(lines[start_idx:], start_idx + 1):
            line = line.strip()
            if not line:
                continue
            
            # Handle quoted text containing delimiters
            if '"' in line:
                parts = []
                in_quote = False
                current_part = []
                for char in line:
                    if char == '"':
                        in_quote = not in_quote
                    elif char == delimiter and not in_quote:
                        parts.append(''.join(current_part).strip())
                        current_part = []
                    else:
                        current_part.append(char)
                parts.append(''.join(current_part).strip())
            else:
                parts = line.split(delimiter)
            
            # Clean parts
            parts = [part.strip().strip('"').strip("'") for part in parts if part.strip()]
            
            if len(parts) >= 2:
                text = parts[0]
                emotion_str = parts[1]
                
                # Convert emotion to unified system
                emotion = map_emotion_to_unified(emotion_str)
                
                texts.append(text)
                emotions.append(emotion)
        
        emotions = np.array(emotions)
        print(f"✅ Manually loaded {len(texts)} samples")
        return np.array(texts), emotions
    
    except Exception as e:
        print(f"❌ Manual parsing also failed: {e}")
        return np.array([]), np.array([], dtype=int)

# TensorFlow Custom Loss Functions
class FocalWeightedLoss(tf.keras.losses.Loss):
    """TensorFlow version of Focal Weighted Loss"""
    def __init__(self, alpha=None, gamma=2.0, class_weights=None, from_logits=False, name='focal_weighted_loss'):
        super().__init__(name=name)
        self.alpha = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None
        self.gamma = gamma
        self.class_weights = tf.constant(class_weights, dtype=tf.float32) if class_weights is not None else None
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        # Convert to probabilities if logits
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        # Convert sparse labels to one-hot if needed
        if len(y_true.shape) == 1:
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        
        # Get probabilities of true classes
        target_probs = tf.reduce_sum(y_true * y_pred, axis=-1)
        
        # Calculate focal term
        focal_term = tf.pow(1 - target_probs, self.gamma)
        
        # Calculate cross entropy
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1)
            ce_loss = alpha_t * ce_loss
        
        # Apply focal term
        focal_loss = focal_term * ce_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            y_true_labels = tf.argmax(y_true, axis=-1)
            weight_factor = tf.gather(self.class_weights, y_true_labels)
            focal_loss = focal_loss * weight_factor
        
        return focal_loss

# Custom TensorFlow Dataset for TER
class TERDataset(tf.keras.utils.Sequence):
    def __init__(self, texts, labels, tokenizer, max_length=128, batch_size=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Tokenize all texts upfront
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )
        
        self.input_ids = encoded['input_ids']
        self.attention_mask = encoded['attention_mask']
        
    def __len__(self):
        return (len(self.texts) + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.texts))
        
        batch_input_ids = self.input_ids[start_idx:end_idx]
        batch_attention_mask = self.attention_mask[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask
        }, batch_labels

def train_ter_model(epochs=12, batch_size=64, lr=2e-5):
    """Train TER model with TensorFlow and custom loss"""
    ter_model_path = os.path.join(CHECKPOINT_DIR, "ter_best")
    ter_tokenizer_path = os.path.join(CHECKPOINT_DIR, "ter_tokenizer")
    
    # Check for existing checkpoint
    if os.path.exists(ter_model_path) and os.path.exists(ter_tokenizer_path):
        print("✅ Loading existing TER model from checkpoint...")
        tokenizer = MobileBertTokenizer.from_pretrained(ter_tokenizer_path)
        model = TFMobileBertForSequenceClassification.from_pretrained(ter_model_path)
        return model, tokenizer
    
    print("🔄 Training TER model with TensorFlow...")
    
    # Load tokenizer and model
    tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    model = TFMobileBertForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased", num_labels=NUM_CLASSES
    )
    
    # Load data
    print("Loading training data...")
    xtr, ytr = load_text_csv(TEXT_TRAIN_CSV)
    print("Loading validation data...")
    xva, yva = load_text_csv(TEXT_VAL_CSV)
    
    # Check if we got valid data
    if len(xtr) == 0 or len(xva) == 0:
        print("❌ No data loaded, skipping TER training")
        return None, None
    
    print(f"Training: {len(xtr)} samples")
    print(f"Validation: {len(xva)} samples")
    
    # Calculate class weights
    class_counts = np.bincount(ytr, minlength=NUM_CLASSES)
    total_samples = len(ytr)
    class_weights = total_samples / (NUM_CLASSES * class_counts + 1e-8)
    print("📊 Class weights for imbalance handling:", class_weights)
    
    # Create datasets
    train_dataset = TERDataset(xtr, ytr, tokenizer, batch_size=batch_size)
    val_dataset = TERDataset(xva, yva, tokenizer, batch_size=batch_size)
    
    # Compile model with custom loss
    focal_loss = FocalWeightedLoss(
        alpha=class_weights,
        gamma=2.0,
        class_weights=class_weights,
        from_logits=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss,
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            ter_model_path,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
    
    # Train model
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and tokenizer
        model.save_pretrained(ter_model_path)
        tokenizer.save_pretrained(ter_tokenizer_path)
        
        print("✅ TER model training completed successfully!")
        return model, tokenizer
    
    except Exception as e:
        print(f"❌ TER training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def ter_predict_proba(model, tokenizer, texts, max_length=128):
    """TensorFlow version of text emotion prediction"""
    if model is None:
        return np.ones((len(texts), NUM_CLASSES)) / NUM_CLASSES
    
    # Tokenize texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    
    # Predict
    outputs = model(encoded)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1)
    
    return probs.numpy()

# ======================= META-CLASSIFIER =======================

def build_meta_classifier(input_dim=21, num_classes=NUM_CLASSES):
    """Build meta-classifier to combine predictions from all modalities"""
    I = layers.Input(shape=(input_dim,))
    
    # Deep architecture with dropout for regularization
    x = layers.Dense(128, activation='relu')(I)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(I, out)

def train_meta_classifier(ter_model, ter_tokenizer, ser_model, fer_models):
    """Train meta-classifier using predictions from all modalities"""
    meta_checkpoint_path = os.path.join(CHECKPOINT_DIR, "meta_best.keras")
    
    # Check for existing checkpoint
    if os.path.exists(meta_checkpoint_path):
        print("✅ Loading existing meta-classifier from checkpoint...")
        meta_model = build_meta_classifier(input_dim=NUM_CLASSES*3)
        meta_model.load_weights(meta_checkpoint_path)
        return meta_model
    
    print("🔄 Training meta-classifier...")
    
    # Load triplet data
    df = pd.read_csv(TRIPLETS_MANIFEST)
    print("📊 Unique labels in triplet manifest:")
    print(df['label'].value_counts())
    
    # Prepare data
    texts = df['text'].astype(str).tolist()
    waves = df['speech_wav'].tolist()
    imgs = df['face_img'].tolist()
    y_true = df['label'].apply(lambda x: map_emotion_to_unified(x)).values
    
    print("  Getting text predictions...")
    print(f"  Sample text: '{texts[0]}'")
    
    # Get text predictions
    p_text = ter_predict_proba(ter_model, ter_tokenizer, texts)
    
    print("  Getting speech predictions...")
    
    # Get speech predictions
    p_speech = []
    for wav_path in waves:
        p_speech.append(ser_predict_proba(ser_model, [wav_path])[0])
    p_speech = np.array(p_speech)
    
    print("  Getting face predictions...")
    
    # Get face predictions in batches
    p_face = []
    batch_size = 32
    for i in range(0, len(imgs), batch_size):
        if (i // batch_size) % 20 == 0:  # Print every 20 batches
            print(f"  Processing image {i+1}/{len(imgs)}")
        
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
    
    p_face = np.array(p_face)
    print(f"  ✅ Successfully loaded {len(p_face)}/{len(imgs)} images")
    
    # Combine features
    X = np.concatenate([p_text, p_speech, p_face], axis=1)
    print(f"📊 Final dataset shape: X={X.shape}, y={y_true.shape}")
    
    # Check label distribution
    unique_labels, counts = np.unique(y_true, return_counts=True)
    print(f"📊 Label distribution: {counts}")
    
    # Split data
    Xtr, Xte, ytr, yte = train_test_split(X, y_true, test_size=0.2, stratify=y_true, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=SEED)
    
    # Build and compile meta-classifier
    meta_model = build_meta_classifier(input_dim=X.shape[1])
    meta_model.compile(
        optimizer=Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=5,
            factor=0.5,
            min_lr=1e-6
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            meta_checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
    
    print("🏋️ Training meta-classifier...")
    
    # Train model
    history = meta_model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best weights
    meta_model.load_weights(meta_checkpoint_path)
    
    # Test evaluation
    test_probs = meta_model.predict(Xte, verbose=0)
    test_pred = test_probs.argmax(1)
    test_acc = accuracy_score(yte, test_pred)
    print(f"\n✅ Meta-classifier Test Accuracy: {test_acc:.4f}")
    
    # Save training results
    results = {
        "test_accuracy": float(test_acc),
        "history": {
            "loss": [float(x) for x in history.history.get('loss', [])],
            "accuracy": [float(x) for x in history.history.get('accuracy', [])],
            "val_loss": [float(x) for x in history.history.get('val_loss', [])],
            "val_accuracy": [float(x) for x in history.history.get('val_accuracy', [])]
        }
    }
    save_checkpoint("meta_training_results", results)
    
    return meta_model

def evaluate_meta_classifier(meta_model, ter_bundle, ser_model, fer_models, test_samples=100):
    """Evaluate the meta-classifier on a subset of data"""
    if meta_model is None:
        print("❌ No meta-classifier available for evaluation")
        return
    
    # Load a subset of data for evaluation
    df = pd.read_csv(TRIPLETS_MANIFEST)
    df_sample = df.sample(min(test_samples, len(df)), random_state=SEED)
    
    # Prepare data
    texts = df_sample['text'].astype(str).tolist()
    waves = df_sample['speech_wav'].tolist()
    imgs = df_sample['face_img'].tolist()
    y_true = df_sample['label'].apply(lambda x: map_emotion_to_unified(x)).values
    
    ter_model, ter_tok = ter_bundle
    
    # Get predictions from individual models
    print("🔍 Getting predictions from individual models...")
    
    # Text predictions
    p_text = ter_predict_proba(ter_model, ter_tok, texts)
    
    # Speech predictions
    p_speech = []
    for wav_path in waves:
        p_speech.append(ser_predict_proba(ser_model, [wav_path])[0])
    p_speech = np.array(p_speech)
    
    # Face predictions
    batch = []
    for img_path in imgs:
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
    p_face = fer_predict_proba(fer_models, batch)
    
    # Combine features for meta-classifier
    X_meta = np.concatenate([p_text, p_speech, p_face], axis=1)
    
    # Meta-classifier predictions
    y_pred_meta = meta_model.predict(X_meta, verbose=0).argmax(axis=1)
    
    # Individual model predictions
    y_pred_text = p_text.argmax(axis=1)
    y_pred_speech = p_speech.argmax(axis=1)
    y_pred_face = p_face.argmax(axis=1)
    
    # Calculate accuracies
    acc_meta = accuracy_score(y_true, y_pred_meta)
    acc_text = accuracy_score(y_true, y_pred_text)
    acc_speech = accuracy_score(y_true, y_pred_speech)
    acc_face = accuracy_score(y_true, y_pred_face)
    
    print("\n📊 MODEL PERFORMANCE COMPARISON:")
    print(f"Meta-classifier Accuracy: {acc_meta:.4f}")
    print(f"Text Model Accuracy: {acc_text:.4f}")
    print(f"Speech Model Accuracy: {acc_speech:.4f}")
    print(f"Face Model Accuracy: {acc_face:.4f}")
    
    # Confusion matrix for meta-classifier
    cm = confusion_matrix(y_true, y_pred_meta)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_ORDER, yticklabels=EMOTION_ORDER)
    plt.title('Meta-classifier Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Classification report
    print("\n📈 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred_meta, target_names=EMOTION_ORDER))
    
    return {
        'meta_accuracy': acc_meta,
        'text_accuracy': acc_text,
        'speech_accuracy': acc_speech,
        'face_accuracy': acc_face,
        'confusion_matrix': cm
    }

# ======================= MAIN TRAINING PIPELINE =======================

def train_all_with_master_checkpoint():
    """Master training function with comprehensive checkpointing"""
    print("🚀 Starting complete multimodal training pipeline...")
    
    # Load or create master checkpoint
    master_checkpoint = load_checkpoint("master_training_status") or {
        "fer_complete": False,
        "ser_complete": False,
        "ter_complete": False,
        "meta_complete": False
    }
    
    # 1. Train FER models
    if not master_checkpoint["fer_complete"]:
        print("\n" + "="*50)
        print("🎯 TRAINING FER MODELS")
        print("="*50)
        fer_ensemble = train_fer_models_with_checkpoints()
        master_checkpoint["fer_complete"] = True
        save_checkpoint("master_training_status", master_checkpoint)
    else:
        print("✅ FER training already complete, loading from checkpoints...")
        fer_ensemble = train_fer_models_with_checkpoints()
    
    # 2. Train SER model
    if not master_checkpoint["ser_complete"]:
        print("\n" + "="*50)
        print("🎯 TRAINING SER MODEL")
        print("="*50)
        ser_model, ser_label_encoder = train_ser_model()
        master_checkpoint["ser_complete"] = True
        save_checkpoint("master_training_status", master_checkpoint)
    else:
        print("✅ SER training already complete, loading from checkpoints...")
        ser_model, ser_label_encoder = train_ser_model()
    
    # 3. Train TER model
    if not master_checkpoint["ter_complete"]:
        print("\n" + "="*50)
        print("🎯 TRAINING TER MODEL")
        print("="*50)
        ter_model, ter_tokenizer = train_ter_model(epochs=1, batch_size=64, lr=2e-5)
        master_checkpoint["ter_complete"] = True
        save_checkpoint("master_training_status", master_checkpoint)
    else:
        print("✅ TER training already complete, loading from checkpoints...")
        ter_model, ter_tokenizer = train_ter_model(epochs=1, batch_size=64, lr=2e-5)
    
    # 4. Train Meta-classifier
    if not master_checkpoint["meta_complete"]:
        print("\n" + "="*50)
        print("🎯 TRAINING META-CLASSIFIER")
        print("="*50)
        meta_model = train_meta_classifier(ter_model, ter_tokenizer, ser_model, fer_ensemble)
        master_checkpoint["meta_complete"] = True
        save_checkpoint("master_training_status", master_checkpoint)
    else:
        print("✅ Meta-classifier training already complete, loading from checkpoints...")
        meta_model = train_meta_classifier(ter_model, ter_tokenizer, ser_model, fer_ensemble)
    
    # Final checkpoint update
    master_checkpoint = {
        "fer_complete": True,
        "ser_complete": True,
        "ter_complete": True,
        "meta_complete": True,
        "training_finished": True
    }
    save_checkpoint("master_training_status", master_checkpoint)
    
    # Create final bundle
    bundle = {
        "ter_model": ter_model,
        "ter_tokenizer": ter_tokenizer,
        "ser_model": ser_model,
        "ser_label_encoder": ser_label_encoder,
        "fer_models": fer_ensemble,
        "meta_model": meta_model
    }
    
    # Save bundle metadata
    bundle_metadata = {"classes": EMOTION_ORDER}
    save_checkpoint("model_bundle_metadata", bundle_metadata)
    
    print("\n" + "="*50)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*50)
    
    return bundle

def main():
    """Main execution function"""
    try:
        # Load SER embeddings first to check data availability
        if os.path.exists(os.path.join(SER_FEATURES_DIR,"features.npy")):
            X, y, label_map = load_ser_embeddings()
            print("Unique labels in y:", np.unique(y))
            print("Label mapping:", label_map)
        
        # Execute complete training pipeline
        print("🚀 Starting complete multimodal training pipeline...")
        bundle = train_all_with_master_checkpoint()
        print("🎉 Training pipeline completed successfully!")
        
        # Extract components for evaluation
        ter_model = bundle["ter_model"]
        ter_tokenizer = bundle["ter_tokenizer"]
        ser_model = bundle["ser_model"]
        fer_ensemble = bundle["fer_models"]
        meta_model = bundle["meta_model"]
        
        # Evaluate the model
        print("🧪 Evaluating meta-classifier performance...")
        results = evaluate_meta_classifier(meta_model, (ter_model, ter_tokenizer), ser_model, fer_ensemble)
        
        print("\n🎉 All training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
