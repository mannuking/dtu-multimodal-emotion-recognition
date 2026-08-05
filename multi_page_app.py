# final_fixed_meta_app.py - FINAL VERSION: Fixed TER + No Flickering
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
from tensorflow.keras import layers, Model
import librosa
from datetime import datetime
import sqlite3
import hashlib
import json
import requests
import gc
import time
import threading
from collections import deque

# Configuration
NUM_CLASSES = 7
EMOTION_ORDER = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
TARGET_SR = 16000
IMG_SIZE = (224, 224)
CHECKPOINT_DIR = "model_checkpoints"
SER_FEATURES_DIR = "ser_feature_output"
OFFSET_S = 0.5
DUR_S = 3.0

# X.AI Configuration — key loaded from environment variable, never hardcoded.
# Set XAI_API_KEY in your shell or a .env file (see .env.example).
import os as _os
XAI_API_KEY = _os.environ.get("XAI_API_KEY", "")
XAI_BASE_URL = "https://api.x.ai/v1"

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# TensorFlow setup
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ TensorFlow GPU setup complete")
except:
    print("⚠️  TensorFlow CPU mode")

# Global state management for no flickering
class AppState:
    def __init__(self):
        self.user_id = None
        self.username = None
        self.logged_in = False
        self.last_emotion = 'Neutral'
        self.last_confidence = 0.0
        self.last_emotion_display = ""
        self.last_psych_display = ""
        self.last_ai_display = ""
        self.processing_lock = threading.Lock()
        self.last_update_time = 0
        self.update_debounce = 5.0  # 1 second debounce
        
    def should_update(self):
        current_time = time.time()
        if current_time - self.last_update_time > self.update_debounce:
            self.last_update_time = current_time
            return True
        return False
    
    def update_emotion(self, emotion, confidence, display_text):
        with self.processing_lock:
            self.last_emotion = emotion
            self.last_confidence = confidence
            self.last_emotion_display = display_text
    
    def update_psychology(self, display_text):
        with self.processing_lock:
            self.last_psych_display = display_text
    
    def get_stable_displays(self):
        with self.processing_lock:
            return self.last_emotion_display, self.last_psych_display

app_state = AppState()

# Database
class EmotionDatabase:
    def __init__(self, db_path="emotion_tracker.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            text_input TEXT,
            detected_emotion TEXT,
            confidence REAL,
            psychological_traits TEXT,
            llm_response TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        return hashlib.sha256((password + "emotion_salt").encode()).hexdigest()
    
    def register_user(self, username, password):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            password_hash = self.hash_password(password)
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def authenticate_user(self, username, password):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        password_hash = self.hash_password(password)
        cursor.execute("SELECT id FROM users WHERE username = ? AND password_hash = ?", (username, password_hash))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def add_interaction(self, user_id, text_input, emotion_data, psychological_traits, llm_response):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO interactions (user_id, text_input, detected_emotion, confidence, psychological_traits, llm_response)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, text_input, emotion_data.get('emotion', 'Neutral'), 
              emotion_data.get('confidence', 0.0), json.dumps(psychological_traits), llm_response))
        conn.commit()
        conn.close()
    
    def get_user_interactions(self, user_id, limit=20):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT detected_emotion, confidence, timestamp, psychological_traits
        FROM interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
        ''', (user_id, limit))
        results = cursor.fetchall()
        conn.close()
        return [{'emotion': row[0], 'confidence': row[1], 'timestamp': row[2], 
                'traits': json.loads(row[3]) if row[3] else {}} for row in results]

# Psychology Analysis
class PsychologicalAnalyzer:
    def analyze_traits(self, interactions):
        if len(interactions) < 20:
            return {
                'status': 'insufficient_data',
                'total_interactions': len(interactions),
                'message': f'Need {20 - len(interactions)} more interactions'
            }
        
        emotions = [i['emotion'] for i in interactions]
        emotion_counts = {e: emotions.count(e) for e in EMOTION_ORDER}
        
        negative_emotions = emotion_counts['Angry'] + emotion_counts['Sad'] + emotion_counts['Fear']
        total = len(interactions)
        negative_ratio = negative_emotions / total
        
        traits = {
            'depression': {
                'risk': 'High Risk' if emotion_counts['Sad'] > 8 else 'Moderate Risk' if emotion_counts['Sad'] > 4 else 'No Risk',
                'score': emotion_counts['Sad'],
                'indicators': ['High sadness frequency'] if emotion_counts['Sad'] > 4 else []
            },
            'anxiety': {
                'risk': 'High Risk' if emotion_counts['Fear'] > 6 else 'Moderate Risk' if emotion_counts['Fear'] > 3 else 'No Risk',
                'score': emotion_counts['Fear'],
                'indicators': ['High fear/anxiety frequency'] if emotion_counts['Fear'] > 3 else []
            },
            'burnout': {
                'risk': 'High Risk' if negative_ratio > 0.6 else 'Moderate Risk' if negative_ratio > 0.4 else 'No Risk',
                'score': int(negative_ratio * 10),
                'indicators': ['High negative emotion ratio'] if negative_ratio > 0.4 else []
            }
        }
        
        return {'status': 'analyzed', 'total_interactions': total, 'traits': traits, 'emotion_distribution': emotion_counts}
    
    def get_recommendations(self, traits):
        recommendations = []
        for trait_name, trait_data in traits.items():
            if trait_data['risk'] in ['High Risk', 'Moderate Risk']:
                if trait_name == 'depression':
                    recommendations.append("Consider engaging in mood-lifting activities")
                elif trait_name == 'anxiety':
                    recommendations.append("Practice relaxation techniques and stress management")
                elif trait_name == 'burnout':
                    recommendations.append("Take regular breaks and prioritize self-care")
        return recommendations

# Fixed Meta-Classifier System
class FixedMetaClassifierSystem:
    def __init__(self):
        self.ter_model = None
        self.ter_tokenizer = None
        self.fer_models = []
        self.ser_model = None
        self.meta_model = None
        self.model_status = {'ter': False, 'fer': False, 'ser': False, 'meta': False}
        
        self.load_all_models()
        print("✅ Fixed Meta-Classifier System Initialized!")
    
    def load_ter_model_fixed(self):
        """Fixed TER model loading with position_ids handling"""
        ter_model_path = os.path.join(CHECKPOINT_DIR, "ter_pytorch_best.pt")
        ter_tokenizer_path = os.path.join(CHECKPOINT_DIR, "ter_pytorch_tokenizer")
        
        if not os.path.exists(ter_model_path) or not os.path.exists(ter_tokenizer_path):
            print("❌ TER model files not found!")
            return
        
        try:
            print("📥 Loading PyTorch TER model (with fixes)...")
            
            # Load tokenizer
            self.ter_tokenizer = AutoTokenizer.from_pretrained(ter_tokenizer_path)
            
            # Load model architecture
            self.ter_model = AutoModelForSequenceClassification.from_pretrained(
                ter_tokenizer_path, 
                num_labels=NUM_CLASSES,
                ignore_mismatched_sizes=True
            ).to(device)
            
            # Load state dict with fixes
            state_dict = torch.load(ter_model_path, map_location=device)
            
            # 🔧 FIX: Remove problematic keys
            keys_to_remove = []
            for key in state_dict.keys():
                if 'position_ids' in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                print(f"🔧 Removing problematic key: {key}")
                del state_dict[key]
            
            # Load with strict=False to handle any remaining mismatches
            missing_keys, unexpected_keys = self.ter_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys (will use default): {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys (ignored): {unexpected_keys}")
            
            self.ter_model.eval()
            self.model_status['ter'] = True
            print("✅ PyTorch TER model loaded successfully!")
            
        except Exception as e:
            print(f"❌ TER loading failed: {e}")
    
    def trim_silence(self, audio, thresh_scale=3):
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
    
    def extract_features(self, y, sr=TARGET_SR, n_mfcc=40):
        hop = int(0.010 * sr)
        win = int(0.025 * sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win, hop_length=hop).T
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=win, hop_length=hop).T
        rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop).T
        
        energy = rms ** 2 * win
        prob = energy / (np.sum(energy) + 1e-8)
        entropy = -prob * np.log2(prob + 1e-12)
        
        feats = np.concatenate([zcr, rms, energy, entropy, mfcc], axis=1)
        return feats.flatten()
    
    def load_fer_models(self):
        print("📥 Loading FER models...")
        model_names = ["vgg16_orig", "vgg16_bal", "resnet50_orig", "resnet50_bal"]
        
        for name in model_names:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.keras")
            if os.path.exists(checkpoint_path):
                try:
                    model = tf.keras.models.load_model(checkpoint_path)
                    self.fer_models.append(model)
                    print(f" ✅ Loaded {name}")
                except Exception as e:
                    print(f" ❌ Failed {name}: {e}")
        
        if len(self.fer_models) > 0:
            self.model_status['fer'] = True
            print(f"✅ FER: {len(self.fer_models)} models loaded")
    
    def load_ser_model(self):
        ser_checkpoint_path = os.path.join(CHECKPOINT_DIR, "ser_best.keras")
        features_path = os.path.join(SER_FEATURES_DIR, "features.npy")
        
        if not os.path.exists(ser_checkpoint_path) or not os.path.exists(features_path):
            print("❌ SER model or features not found!")
            return
        
        try:
            print("📥 Loading SER model...")
            f = np.load(features_path)
            
            # Build SER architecture
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
            
            self.ser_model = Model(I, out)
            self.ser_model.load_weights(ser_checkpoint_path)
            self.model_status['ser'] = True
            print("✅ SER model loaded")
            
        except Exception as e:
            print(f"❌ SER loading failed: {e}")
    
    def load_meta_classifier(self):
        meta_checkpoint_path = os.path.join(CHECKPOINT_DIR, "meta_hybrid_best.keras")
        if not os.path.exists(meta_checkpoint_path):
            print("❌ Meta-classifier not found!")
            return
        
        try:
            print("📥 Loading Meta-classifier...")
            self.meta_model = tf.keras.models.load_model(meta_checkpoint_path)
            self.model_status['meta'] = True
            print("✅ Meta-classifier loaded successfully!")
        except Exception as e:
            print(f"❌ Meta-classifier loading failed: {e}")
    
    def load_all_models(self):
        print("🔄 Loading all models with fixes...")
        self.load_ter_model_fixed()  # 🔧 Fixed version
        self.load_fer_models()
        self.load_ser_model()
        self.load_meta_classifier()
        gc.collect()
    
    @torch.no_grad()
    def predict_ter(self, text):
        if not self.model_status['ter'] or not text:
            return self.predict_text_fallback(text)
        
        try:
            tokens = self.ter_tokenizer([str(text)], padding=True, truncation=True, max_length=128, return_tensors='pt')
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            logits = self.ter_model(**tokens).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Clean up
            del tokens, logits
            if device.type == 'mps':
                torch.mps.empty_cache()
            
            return probs
            
        except Exception as e:
            print(f"TER error: {e}")
            return self.predict_text_fallback(text)
    
    def predict_text_fallback(self, text):
        if not text:
            probs = np.zeros(NUM_CLASSES)
            probs[6] = 1.0  # Neutral
            return probs
        
        text = str(text).lower()
        probs = np.zeros(NUM_CLASSES)
        
        if any(word in text for word in ['happy', 'joy', 'excited', 'great', 'fantastic', 'amazing']):
            probs[3] = 0.8  # Happy
            probs[6] = 0.2
        elif any(word in text for word in ['sad', 'depressed', 'down', 'upset', 'terrible']):
            probs[4] = 0.8  # Sad
            probs[6] = 0.2
        elif any(word in text for word in ['angry', 'mad', 'furious', 'annoyed', 'hate']):
            probs[0] = 0.8  # Angry
            probs[6] = 0.2
        elif any(word in text for word in ['scared', 'afraid', 'worried', 'anxious']):
            probs[2] = 0.8  # Fear
            probs[6] = 0.2
        else:
            probs[6] = 1.0  # Neutral
            
        return probs
    
    def predict_fer(self, image):
        if not self.model_status['fer'] or image is None:
            return np.ones(NUM_CLASSES) / NUM_CLASSES
        
        try:
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image.astype('uint8')).convert('RGB')
            elif hasattr(image, 'convert'):
                img = image.convert('RGB')
            else:
                return np.ones(NUM_CLASSES) / NUM_CLASSES
            
            img = img.resize(IMG_SIZE)
            img_array = np.asarray(img).astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            preds = [m.predict(img_batch, verbose=0) for m in self.fer_models]
            return np.mean(preds, axis=0)[0]
            
        except Exception as e:
            print(f"FER error: {e}")
            return np.ones(NUM_CLASSES) / NUM_CLASSES
    
    def predict_ser(self, audio_array):
        if not self.model_status['ser'] or audio_array is None:
            return np.ones(NUM_CLASSES) / NUM_CLASSES
        
        try:
            if len(audio_array.shape) == 2:
                audio_array = np.mean(audio_array, axis=1)
            
            y = self.trim_silence(audio_array.astype(np.float32))
            y = y[int(OFFSET_S * TARGET_SR):]
            target = int(DUR_S * TARGET_SR)
            
            if len(y) > target:
                y = y[:target]
            else:
                y = np.pad(y, (0, target - len(y)))
            
            features = self.extract_features(y)
            X = features.reshape(1, -1, 1)
            
            return self.ser_model.predict(X, verbose=0)[0]
            
        except Exception as e:
            print(f"SER error: {e}")
            return np.ones(NUM_CLASSES) / NUM_CLASSES
    
    def predict_multimodal(self, text, image=None, audio=None):
        start_time = datetime.now()
        
        try:
            # Get individual predictions
            ter_probs = self.predict_ter(text)
            fer_probs = self.predict_fer(image)
            ser_probs = self.predict_ser(audio)
            
            # Meta-classifier prediction
            if self.model_status['meta']:
                try:
                    combined_features = np.concatenate([ter_probs, ser_probs, fer_probs])
                    combined_features = combined_features.reshape(1, -1)
                    
                    meta_probs = self.meta_model.predict(combined_features, verbose=0)[0]
                    model_type = "meta_classifier"
                except Exception as e:
                    print(f"Meta-classifier error: {e}")
                    weights = [0.4, 0.3, 0.3]  # TER, FER, SER
                    meta_probs = weights[0] * ter_probs + weights[1] * fer_probs + weights[2] * ser_probs
                    model_type = "ensemble_fallback"
            else:
                weights = [0.4, 0.3, 0.3]
                meta_probs = weights[0] * ter_probs + weights[1] * fer_probs + weights[2] * ser_probs
                model_type = "ensemble_fallback"
            
            final_emotion = EMOTION_ORDER[np.argmax(meta_probs)]
            final_confidence = float(np.max(meta_probs))
            
            inference_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'TER': ter_probs,
                'TER_emotion': EMOTION_ORDER[np.argmax(ter_probs)],
                'TER_confidence': float(np.max(ter_probs)),
                'FER': fer_probs,
                'FER_emotion': EMOTION_ORDER[np.argmax(fer_probs)],
                'FER_confidence': float(np.max(fer_probs)),
                'SER': ser_probs,
                'SER_emotion': EMOTION_ORDER[np.argmax(ser_probs)],
                'SER_confidence': float(np.max(ser_probs)),
                'META': meta_probs,
                'META_emotion': final_emotion,
                'META_confidence': final_confidence,
                'inference_time': inference_time,
                'model_type': model_type,
                'model_status': self.model_status.copy()
            }
            
        except Exception as e:
            print(f"Multimodal prediction error: {e}")
            neutral_probs = np.ones(NUM_CLASSES) / NUM_CLASSES
            return {
                'TER': neutral_probs, 'TER_emotion': 'Neutral', 'TER_confidence': 1.0/NUM_CLASSES,
                'FER': neutral_probs, 'FER_emotion': 'Neutral', 'FER_confidence': 1.0/NUM_CLASSES,
                'SER': neutral_probs, 'SER_emotion': 'Neutral', 'SER_confidence': 1.0/NUM_CLASSES,
                'META': neutral_probs, 'META_emotion': 'Neutral', 'META_confidence': 1.0/NUM_CLASSES,
                'inference_time': 0.001, 'model_type': 'fallback',
                'model_status': {'ter': False, 'fer': False, 'ser': False, 'meta': False}
            }

# Initialize components
db = EmotionDatabase()
psychology = PsychologicalAnalyzer()
recognizer = FixedMetaClassifierSystem()

def call_xai_api(text_input, detected_emotion, psychological_traits):
    try:
        psychological_context = ""
        if psychological_traits and psychological_traits.get('status') == 'analyzed':
            traits = psychological_traits.get('traits', {})
            high_risk_traits = [name for name, data in traits.items() 
                              if data.get('risk') in ['High Risk', 'Moderate Risk']]
            if high_risk_traits:
                psychological_context = f"Note: Student shows signs of {', '.join(high_risk_traits)}. Please be extra supportive."
        
        system_prompt = f"""You are a caring AI tutor. The student is currently feeling {detected_emotion.lower()}. {psychological_context}

Respond with empathy and helpful guidance while being mindful of their emotional state."""
        
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {XAI_API_KEY}"}
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input}
            ],
            "model": "grok-beta",
            "stream": False,
            "temperature": 0.8,
            "max_tokens": 300
        }
        
        response = requests.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"I can see you're feeling {detected_emotion.lower()}. While I'm having technical difficulties, I want you to know your feelings are valid and I'm here to support you."
            
    except Exception as e:
        return f"I understand you're feeling {detected_emotion.lower()}. Even though I'm experiencing some technical issues, I want to help. Your question about '{text_input}' is important to me."

def authenticate_user(username, password):
    if not username or not password:
        return "Please enter username and password", gr.update(visible=True), gr.update(visible=False)
    
    user_id = db.authenticate_user(username, password)
    if user_id:
        app_state.user_id = user_id
        app_state.username = username
        app_state.logged_in = True
        return f"✅ Welcome back, {username}!", gr.update(visible=False), gr.update(visible=True)
    return "❌ Invalid username or password", gr.update(visible=True), gr.update(visible=False)

def register_user(username, password):
    if not username or not password:
        return "Please enter username and password"
    if len(password) < 6:
        return "Password must be at least 6 characters"
    if db.register_user(username, password):
        return "✅ Registration successful! You can now login."
    return "❌ Username already exists. Please choose another."

def process_real_time_emotion_stable(text, image, audio):
    """🔧 FIXED: No flickering version with debouncing"""
    if not app_state.logged_in:
        return app_state.get_stable_displays()
    
    # 🔧 Debounce rapid updates
    if not app_state.should_update():
        return app_state.get_stable_displays()
    
    try:
        # Process audio
        if audio is not None:
            try:
                sample_rate, audio_data = audio
                if len(audio_data.shape) == 2:
                    audio_data = np.mean(audio_data, axis=1)
                audio_data = audio_data.astype(np.float32)
                
                if sample_rate != TARGET_SR:
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SR)
            except:
                audio_data = None
        else:
            audio_data = None
        
        # Get prediction
        results = recognizer.predict_multimodal(text, image, audio_data)
        
        # Get psychological analysis
        interactions = db.get_user_interactions(app_state.user_id)
        psych_analysis = psychology.analyze_traits(interactions)
        
        # Format displays
        emotion_output = f"""**🧠 META-CLASSIFIER PREDICTION: {results['META_emotion']}** ({results['META_confidence']:.1%})

**Individual Model Contributions:**
📝 Text Analysis: {results['TER_emotion']} ({results['TER_confidence']:.1%}) {'✅' if recognizer.model_status['ter'] else '🔄'}
📷 Facial Analysis: {results['FER_emotion']} ({results['FER_confidence']:.1%}) {'✅' if recognizer.model_status['fer'] else '❌'}
🎤 Speech Analysis: {results['SER_emotion']} ({results['SER_confidence']:.1%}) {'✅' if recognizer.model_status['ser'] else '❌'}

⚡ Meta-Classifier Inference: {results['inference_time']:.3f}s
🎯 Model Status: TER: {'✅' if results['model_status']['ter'] else '🔄'}, FER: {'✅' if results['model_status']['fer'] else '❌'}, SER: {'✅' if results['model_status']['ser'] else '❌'}, META: {'✅' if results['model_status']['meta'] else '❌'}
🔬 Model Type: {results['model_type'].upper()}"""
        
        if psych_analysis['status'] == 'analyzed' and psych_analysis['total_interactions'] >= 20:
            traits = psych_analysis['traits']
            psych_output = f"""**Psychological Analysis** ({psych_analysis['total_interactions']} interactions)

🧠 **Depression Risk**: {traits['depression']['risk']}
   Score: {traits['depression']['score']}/20 interactions

😰 **Anxiety Risk**: {traits['anxiety']['risk']}
   Score: {traits['anxiety']['score']}/20 interactions

😴 **Burnout Risk**: {traits['burnout']['risk']}
   Score: {traits['burnout']['score']}/10 scale"""
            
            recommendations = psychology.get_recommendations(traits)
            if recommendations:
                psych_output += f"\n\n**💡 Recommendations:**"
                for i, rec in enumerate(recommendations[:3], 1):
                    psych_output += f"\n{i}. {rec}"
        else:
            remaining = max(0, 20 - psych_analysis['total_interactions'])
            psych_output = f"""**Psychological Analysis**

Need {remaining} more interactions for complete analysis.

Current interactions: {psych_analysis['total_interactions']}/20

**Current Meta-Classifier Emotion:** {results['META_emotion']} ({results['META_confidence']:.1%})"""
        
        # 🔧 Update state atomically
        app_state.update_emotion(results['META_emotion'], results['META_confidence'], emotion_output)
        app_state.update_psychology(psych_output)
        
        return app_state.get_stable_displays()
        
    except Exception as e:
        print(f"💥 Processing error: {e}")
        return app_state.get_stable_displays()

def submit_to_ai(user_question, image_input, audio_input):
    if not app_state.logged_in:
        return "Please login first"
    if not user_question or not user_question.strip():
        return "Please enter a question or message"
    
    # 1. Run meta-classifier on the submitted input (text, image, audio)
    #    This ensures the emotion is tied to the user's actual question, not any real-time state.
    if audio_input is not None:
        try:
            sample_rate, audio_data = audio_input
            if len(audio_data.shape) == 2:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32)
            if sample_rate != TARGET_SR:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SR)
        except Exception:
            audio_data = None
    else:
        audio_data = None

    # Get meta-classifier prediction for this submission
    results = recognizer.predict_multimodal(user_question, image_input, audio_data)
    last_emotion = results['META_emotion']
    last_confidence = results['META_confidence']

    # 2. Get psychological analysis
    interactions = db.get_user_interactions(app_state.user_id)
    psych_analysis = psychology.analyze_traits(interactions)

    # 3. Call the AI API with the emotion from this submission
    ai_response = call_xai_api(user_question, last_emotion, psych_analysis)

    # 4. Save the interaction to the database
    if app_state.user_id:
        db.add_interaction(
            app_state.user_id,
            user_question,
            {'emotion': last_emotion, 'confidence': last_confidence},
            psych_analysis,
            ai_response
        )

    # 5. Format and return the response
    formatted_response = f"""**🤖 AI Tutor Response (Meta-Classifier Enhanced)**
*Current Emotion: {last_emotion} ({last_confidence:.1%})*

{ai_response}

---
*Powered by Meta-Classifier emotion recognition + Psychological analysis*"""

    app_state.last_ai_display = formatted_response
    return formatted_response

def logout_user():
    app_state.logged_in = False
    app_state.user_id = None
    app_state.username = None
    app_state.last_emotion = 'Neutral'
    app_state.last_confidence = 0.0
    app_state.last_emotion_display = ""
    app_state.last_psych_display = ""
    app_state.last_ai_display = ""
    
    return ("Logged out successfully", "", "", gr.update(visible=True), gr.update(visible=False), "", "", "", "")

def create_final_fixed_app():
    css_styles = """
        body, .gradio-container {
        background: #181c24 !important;
        color: #e0e6f0 !important;
        font-family: 'JetBrains Mono', 'Fira Mono', 'Menlo', monospace;
    }
    .meta-header {
        background: linear-gradient(90deg, #0f2027 0%, #2c5364 100%);
        color: #fff;
        padding: 32px 0 18px 0;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 18px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.18);
        font-size: 2.1rem;
        letter-spacing: 1.5px;
    }
    .gr-button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%) !important;
        color: #fff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 1px;
    }
    .gr-textbox, .gr-image, .gr-audio {
        background: rgba(30,34,44,0.95) !important;
        border-radius: 10px !important;
        border: 1.5px solid #232a3a !important;
    }
    .output-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: #00eaff;
        background: rgba(20,24,34,0.95);
        border-radius: 10px;
        padding: 12px;
        margin-top: 8px;
        min-height: 80px;
    }
    """
    
    with gr.Blocks(css=css_styles, title="Meta-Classifier") as app:
        gr.HTML("""
        <div class="meta-header">
            <span style="font-size:2.5rem;">🧬</span><br>
            <b>META-CLASSIFIER</b> <span style="color:#00eaff;">EMOTION AI</span>
        </div>
        """)
        with gr.Column(visible=True) as login_section:
            username = gr.Textbox(label="Username", placeholder="Username")
            password = gr.Textbox(label="Password", type="password", placeholder="Password")
            with gr.Row():
                login_btn = gr.Button("Login")
                register_btn = gr.Button("Register")
            login_status = gr.Textbox(label="", interactive=False, show_label=False)
        with gr.Column(visible=False) as main_section:
            with gr.Row():
                text_input = gr.Textbox(label="Type or Speak", placeholder="How are you feeling?", lines=2)
                camera_input = gr.Image(source="webcam", streaming=True, label="Face", height=180)
                audio_input = gr.Audio(source="microphone", type="numpy", label="Voice", streaming=True)
            with gr.Row():
                ask_ai_btn = gr.Button("Ask AI")
                logout_btn = gr.Button("Logout")
            emotion_display = gr.Textbox(label="Emotion", lines=2, interactive=False, elem_classes=["output-text"])
            psychological_display = gr.Textbox(label="Psychology", lines=2, interactive=False, elem_classes=["output-text"])
            ai_display = gr.Textbox(label="AI Response", lines=3, interactive=False, elem_classes=["output-text"])
        
        # 🔧 Fixed Event Handlers with debouncing
        login_btn.click(authenticate_user, inputs=[username, password], outputs=[login_status, login_section, main_section])
        password.submit(authenticate_user, inputs=[username, password], outputs=[login_status, login_section, main_section])
        register_btn.click(register_user, inputs=[username, password], outputs=[login_status])
        
        # 🔧 FIXED: Stable processing with debouncing
        text_input.change(process_real_time_emotion_stable, inputs=[text_input, camera_input, audio_input], outputs=[emotion_display, psychological_display])
        camera_input.change(process_real_time_emotion_stable, inputs=[text_input, camera_input, audio_input], outputs=[emotion_display, psychological_display])
        
        ask_ai_btn.click(submit_to_ai, inputs=[text_input], outputs=[ai_display])
        text_input.submit(submit_to_ai, inputs=[text_input], outputs=[ai_display])
        
        logout_btn.click(logout_user, outputs=[login_status, username, password, login_section, main_section, emotion_display, psychological_display, ai_display, text_input])
    
    return app

if __name__ == "__main__":
    print("🚀 LAUNCHING FIXED FINAL META-CLASSIFIER SYSTEM")
    print("=" * 90)
    print("🔧 FIXES IMPLEMENTED:")
    print("   ✅ TER Model Loading: Fixed position_ids key removal")
    print("   ✅ No Flickering: Debouncing + State management")  
    print("   ✅ Thread Safety: Atomic updates with locks")
    print("   ✅ Error Handling: Graceful fallbacks")
    print("=" * 90)
    print("📊 COMPONENT STATUS:")
    print(f"   🎭 TER: {'✅ FIXED' if recognizer.model_status['ter'] else '🔄 Fallback Active'}")
    print(f"   📷 FER: {'✅ Loaded (' + str(len(recognizer.fer_models)) + ' models)' if recognizer.model_status['fer'] else '❌ Missing'}")
    print(f"   🎤 SER: {'✅ Loaded' if recognizer.model_status['ser'] else '❌ Missing'}")
    print(f"   🧠 META: {'✅ Loaded' if recognizer.model_status['meta'] else '❌ Missing'}")
    print("=" * 90)
    
    app = create_final_fixed_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        debug=False
    )
