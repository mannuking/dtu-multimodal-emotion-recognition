# train_ser_tensorflow.py - Train SER with TensorFlow (your exact architecture)

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from keras import layers, Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import librosa
from gpu_config import *
from feature_utils import ensure_features_exist, EMOTION_ORDER_LOWER

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
    
    model = build_ser_1d_cnn((X.shape[1], 1))
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.1, min_lr=1e-5),
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        ModelCheckpoint(ser_checkpoint_path, save_best_only=True, monitor='val_accuracy', verbose=1)
    ]
    
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=50, batch_size=32, callbacks=callbacks, verbose=1)
    
    with open(ser_encoder_path, 'wb') as f:
        pickle.dump(labelmap, f)
    
    print("✅ SER model training complete!")

if __name__ == "__main__":
    train_ser_model()
