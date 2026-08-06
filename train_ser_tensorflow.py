# train_ser_tensorflow.py - Train SER with TensorFlow (your exact architecture)

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from keras import layers, Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from train_utils import focal_weighted_cce, class_weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import librosa
from gpu_config import *
from gpu_runtime import enable_tf_perf, enable_mixed_precision, set_seed, global_batch_size
from feature_utils import ensure_features_exist, EMOTION_ORDER_LOWER

# Initialize perf runtime + strategy (mixed precision, XLA, multi-GPU)
STRATEGY = enable_tf_perf(mixed_precision=False)  # defer — SER enables it inside scope
set_seed(SEED)

tf.random.set_seed(SEED)

def load_ser_embeddings():
    # Use the cached features if present, otherwise compute on-the-fly from
    # the audio manifest and cache for next run.
    feats, labels = ensure_features_exist(
        features_path=os.path.join(SER_FEATURES_DIR, "features.npy"),
        labels_path=os.path.join(SER_FEATURES_DIR, "labels.npy"),
        manifest_csv="combined_ser_dataset/metadata.csv",
    )
    # labels is int array — convert back to string for the rest of the pipeline
    label_map = {e: i for i, e in enumerate(EMOTION_ORDER_LOWER)}
    l_str = np.array([EMOTION_ORDER_LOWER[int(lbl)] if int(lbl) in range(len(EMOTION_ORDER_LOWER)) else "neutral" for lbl in labels])
    mask = np.array([lbl in label_map for lbl in l_str])
    f, l = feats[mask], l_str[mask]
    y = np.array([label_map[lbl] for lbl in l])
    X = f.reshape(f.shape[0], f.shape[1], 1)

    le = LabelEncoder()
    le.classes_ = np.array(EMOTION_ORDER_LOWER)
    return X, y, le

def build_ser_1d_cnn(input_shape, num_classes=NUM_CLASSES):
    I = layers.Input(shape=input_shape)
    
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
    out = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(I, out)

def train_ser_model():
    ser_checkpoint_path = os.path.join(CHECKPOINT_DIR, "ser_best.keras")
    ser_encoder_path = os.path.join(CHECKPOINT_DIR, "ser_label_encoder.pkl")
    
    if os.path.exists(ser_checkpoint_path) and os.path.exists(ser_encoder_path):
        print("✅ SER model already trained!")
        return
    
    print("🔄 Training SER model...")
    
    X, y, labelmap = load_ser_embeddings()
    
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.111, stratify=ytr, random_state=SEED)

    # Paper Sec 5.1: focal weighted categorical cross-entropy with class-frequency alpha
    alpha = np.array([class_weights(y).get(int(c), 1.0) for c in range(NUM_CLASSES)], dtype=np.float32)

    # Multi-GPU: build + compile inside strategy.scope(). MirroredStrategy
    # splits batches across GPUs automatically.
    with STRATEGY.scope():
        # SER trains from scratch — safe to use mixed precision here.
        enable_mixed_precision()
        model = build_ser_1d_cnn((X.shape[1], 1))
        model.compile(
            optimizer=Adam(1e-3),
            loss=focal_weighted_cce(alpha, gamma=2.0),
            metrics=['accuracy'],
        )

    callbacks = [
        # Paper: ReduceLROnPlateau patience=2, factor=0.5
        ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, min_lr=1e-5),
        # Paper: early stopping patience 4-8
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        ModelCheckpoint(ser_checkpoint_path, save_best_only=True, monitor='val_accuracy', verbose=1),
    ]

    # Paper: batch size 64, epochs 30-50. Audio augmentation (noise + gain) applied
    # on the tf.data pipeline. Pre-extracted features are augmented in-graph.
    # Paper: batch size 64 per replica; multi-GPU auto-scales to 64*N.
    print("   audio augmentation enabled (noise + gain, paper Sec 5.1)")
    per_replica_bs = 64
    batch_size = global_batch_size(per_replica_bs, STRATEGY)
    print(f"   per-replica batch: {per_replica_bs}  ×  {STRATEGY.num_replicas_in_sync} GPUs  =  {batch_size} global")
    def aug(x, y):
        noise = tf.random.normal(tf.shape(x), stddev=0.005, dtype=x.dtype)
        gain = tf.random.uniform([], 0.85, 1.15, dtype=x.dtype)
        return x * gain + noise, y
    train_ds = (
        tf.data.Dataset.from_tensor_slices((Xtr, ytr))
        .shuffle(8192)
        .batch(batch_size)
        .map(aug, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.fit(
        train_ds,
        validation_data=(Xva, yva),
        epochs=50,
        callbacks=callbacks,
        verbose=1,
    )
    
    with open(ser_encoder_path, 'wb') as f:
        pickle.dump(labelmap, f)
    
    print("✅ SER model training complete!")

if __name__ == "__main__":
    train_ser_model()
