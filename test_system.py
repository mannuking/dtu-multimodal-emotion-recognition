# proper_meta_inference.py - Complete inference system matching your training setup
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
from PIL import Image
import pickle
import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras import layers, Model
import librosa
from datetime import datetime
import gc

# Configuration (create this config.py based on your training setup)
NUM_CLASSES = 7
EMOTION_ORDER = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
TARGET_SR = 16000
IMG_SIZE = (224, 224)
CHECKPOINT_DIR = "model_checkpoints"
SER_FEATURES_DIR = "ser_feature_output"
OFFSET_S = 0.5
DUR_S = 3.0
SEED = 42

# GPU setup
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"🔥 Using device: {device}")

# TensorFlow GPU setup
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ TensorFlow GPU setup complete")
except:
    print("⚠️  TensorFlow CPU mode")

class CompleteMetaClassifier:
    def __init__(self):
        self.ter_model = None
        self.ter_tokenizer = None
        self.fer_models = []
        self.ser_model = None
        self.meta_model = None
        
        self.model_status = {
            'ter': False,
            'fer': False, 
            'ser': False,
            'meta': False
        }
        
        self.load_all_models()
        print("✅ Complete Meta-classifier System Initialized!")
    
    def map_emotion_to_unified(self, emotion_str, source=None):
        """Map emotion string to unified format"""
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
    
    @torch.no_grad()
    def load_ter_pytorch(self):
        """Load trained PyTorch TER model (exact match to training)"""
        ter_model_path = os.path.join(CHECKPOINT_DIR, "ter_pytorch_best.pt")
        ter_tokenizer_path = os.path.join(CHECKPOINT_DIR, "ter_pytorch_tokenizer")
        
        if not os.path.exists(ter_model_path):
            print("❌ TER model not found!")
            return
        
        try:
            print("📥 Loading PyTorch TER model...")
            self.ter_tokenizer = MobileBertTokenizer.from_pretrained(ter_tokenizer_path)
            self.ter_model = MobileBertForSequenceClassification.from_pretrained(
                ter_tokenizer_path, num_labels=NUM_CLASSES
            ).to(device)
            self.ter_model.load_state_dict(torch.load(ter_model_path, map_location=device))
            self.ter_model.eval()
            self.model_status['ter'] = True
            print("✅ PyTorch TER model loaded")
        except Exception as e:
            print(f"❌ TER loading failed: {e}")
    
    def load_fer_tensorflow(self):
        """Load trained TensorFlow FER ensemble (exact match to training)"""
        model_names = ["vgg16_orig", "vgg16_bal", "resnet50_orig", "resnet50_bal"]
        print("📥 Loading TensorFlow FER models...")
        
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
    
    def trim_silence(self, audio, thresh_scale=3):
        """Trim silence from audio (exact match to training)"""
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
        """Extract audio features (exact match to training)"""
        hop = int(0.010 * sr)  # 10ms
        win = int(0.025 * sr)  # 25ms
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=win, hop_length=hop).T
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=win, hop_length=hop).T
        rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop).T
        
        energy = rms ** 2 * win
        prob = energy / (np.sum(energy) + 1e-8)
        entropy = -prob * np.log2(prob + 1e-12)
        
        feats = np.concatenate([zcr, rms, energy, entropy, mfcc], axis=1)
        return feats.flatten()
    
    def load_ser_tensorflow(self):
        """Load trained TensorFlow SER model (exact match to training)"""
        ser_checkpoint_path = os.path.join(CHECKPOINT_DIR, "ser_best.keras")
        
        if not os.path.exists(ser_checkpoint_path):
            print("❌ SER model not found!")
            return
        
        # Check if we have feature files to determine input shape
        features_path = os.path.join(SER_FEATURES_DIR, "features.npy")
        if not os.path.exists(features_path):
            print("❌ SER features not found for shape inference!")
            return
        
        try:
            print("📥 Loading TensorFlow SER model...")
            
            # Load dummy features to get input shape
            f = np.load(features_path)
            
            # Build model architecture (exact match to training)
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
            print("✅ TensorFlow SER model loaded")
            
        except Exception as e:
            print(f"❌ SER loading failed: {e}")
    
    def load_meta_classifier(self):
        """Load trained meta-classifier (exact match to training)"""
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
        """Load all models in proper order"""
        print("🔄 Loading all models...")
        self.load_ter_pytorch()
        self.load_fer_tensorflow()
        self.load_ser_tensorflow()
        self.load_meta_classifier()
        
        # Memory cleanup
        gc.collect()
        if device.type in ['cuda', 'mps']:
            try:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
            except:
                pass
        
        print("\n📊 Model Loading Summary:")
        print(f"  TER: {'✅' if self.model_status['ter'] else '❌'}")
        print(f"  FER: {'✅' if self.model_status['fer'] else '❌'}")
        print(f"  SER: {'✅' if self.model_status['ser'] else '❌'}")
        print(f"  META: {'✅' if self.model_status['meta'] else '❌'}")
    
    @torch.no_grad()
    def ter_predict_proba(self, texts, maxlen=128):
        """TER prediction (exact match to training)"""
        if not self.model_status['ter']:
            return np.ones((len(texts), NUM_CLASSES)) / NUM_CLASSES
        
        try:
            texts_cleaned = []
            for text in texts:
                if pd.isna(text) or text is None:
                    text = "neutral"
                else:
                    text = str(text).strip()
                texts_cleaned.append(text)
            
            tok = self.ter_tokenizer(texts_cleaned, padding=True, truncation=True, max_length=maxlen, return_tensors='pt')
            tok = {k: v.to(device) for k, v in tok.items()}
            
            logits = self.ter_model(**tok).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            return probs
            
        except Exception as e:
            print(f"TER prediction error: {e}")
            return np.ones((len(texts), NUM_CLASSES)) / NUM_CLASSES
    
    def fer_predict_proba(self, img_array_batch):
        """FER ensemble prediction (exact match to training)"""
        if not self.model_status['fer']:
            return np.ones((len(img_array_batch), NUM_CLASSES)) / NUM_CLASSES
        
        try:
            preds = [m.predict(img_array_batch, verbose=0) for m in self.fer_models]
            return np.mean(preds, axis=0)
        except Exception as e:
            print(f"FER prediction error: {e}")
            return np.ones((len(img_array_batch), NUM_CLASSES)) / NUM_CLASSES
    
    def preprocess_audio_from_array(self, audio_array, sr=TARGET_SR):
        """Preprocess audio array for SER (match training preprocessing)"""
        try:
            if audio_array is None or len(audio_array) == 0:
                return None
            
            # Resample if needed
            if sr != TARGET_SR:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
            
            # Trim silence
            y = self.trim_silence(audio_array)
            
            # Apply offset and duration (same as training)
            y = y[int(OFFSET_S * TARGET_SR):]
            target = int(DUR_S * TARGET_SR)
            
            if len(y) > target:
                y = y[:target]
            else:
                y = np.pad(y, (0, target - len(y)))
            
            return y
            
        except Exception as e:
            print(f"Audio preprocessing error: {e}")
            return None
    
    def ser_predict_proba_from_array(self, audio_arrays):
        """SER prediction from audio arrays (exact match to training)"""
        if not self.model_status['ser']:
            return np.ones((len(audio_arrays), NUM_CLASSES)) / NUM_CLASSES
        
        try:
            features = []
            for audio_array in audio_arrays:
                if audio_array is not None:
                    y = self.preprocess_audio_from_array(audio_array)
                    if y is not None:
                        feat = self.extract_features(y)
                    else:
                        feat = np.zeros(self.ser_model.input_shape[1])
                else:
                    feat = np.zeros(self.ser_model.input_shape[1])
                features.append(feat)
            
            X = np.array(features).reshape(len(features), -1, 1)
            return self.ser_model.predict(X, verbose=0)
            
        except Exception as e:
            print(f"SER prediction error: {e}")
            return np.ones((len(audio_arrays), NUM_CLASSES)) / NUM_CLASSES
    
    def predict_multimodal(self, text, image=None, audio_array=None):
        """Complete multimodal prediction using meta-classifier"""
        start_time = datetime.now()
        
        try:
            # Prepare inputs as lists for batch processing
            texts = [text] if isinstance(text, str) else text
            
            # TER prediction
            ter_probs = self.ter_predict_proba(texts)
            
            # FER prediction
            if image is not None:
                if isinstance(image, np.ndarray):
                    img = Image.fromarray(image.astype('uint8')).convert('RGB')
                elif hasattr(image, 'convert'):
                    img = image.convert('RGB')
                else:
                    img = Image.new('RGB', IMG_SIZE)
                
                img = img.resize(IMG_SIZE)
                img_array = np.asarray(img).astype(np.float32) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)
                fer_probs = self.fer_predict_proba(img_batch)
            else:
                fer_probs = np.ones((1, NUM_CLASSES)) / NUM_CLASSES
            
            # SER prediction
            if audio_array is not None:
                ser_probs = self.ser_predict_proba_from_array([audio_array])
            else:
                ser_probs = np.ones((1, NUM_CLASSES)) / NUM_CLASSES
            
            # Meta-classifier prediction
            if self.model_status['meta']:
                # Combine features: TER + SER + FER (same order as training)
                combined_features = np.concatenate([ter_probs[0], ser_probs[0], fer_probs[0]])
                combined_features = combined_features.reshape(1, -1)
                
                meta_probs = self.meta_model.predict(combined_features, verbose=0)[0]
                model_type = "meta_classifier"
            else:
                # Fallback ensemble
                weights = [0.5, 0.3, 0.2]  # TER, FER, SER
                meta_probs = (weights[0] * ter_probs[0] + 
                             weights[1] * fer_probs[0] + 
                             weights[2] * ser_probs[0])
                model_type = "ensemble_fallback"
            
            final_emotion = EMOTION_ORDER[np.argmax(meta_probs)]
            final_confidence = float(np.max(meta_probs))
            
            inference_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'TER': ter_probs[0],
                'TER_emotion': EMOTION_ORDER[np.argmax(ter_probs[0])],
                'TER_confidence': float(np.max(ter_probs[0])),
                'FER': fer_probs[0],
                'FER_emotion': EMOTION_ORDER[np.argmax(fer_probs[0])],
                'FER_confidence': float(np.max(fer_probs[0])),
                'SER': ser_probs[0],
                'SER_emotion': EMOTION_ORDER[np.argmax(ser_probs[0])],
                'SER_confidence': float(np.max(ser_probs[0])),
                'META': meta_probs,
                'META_emotion': final_emotion,
                'META_confidence': final_confidence,
                'inference_time': inference_time,
                'model_type': model_type,
                'model_status': self.model_status.copy()
            }
            
        except Exception as e:
            print(f"Multimodal prediction error: {e}")
            
            # Ultimate fallback
            neutral_probs = np.ones(NUM_CLASSES) / NUM_CLASSES
            return {
                'TER': neutral_probs, 'TER_emotion': 'Neutral', 'TER_confidence': 1.0/NUM_CLASSES,
                'FER': neutral_probs, 'FER_emotion': 'Neutral', 'FER_confidence': 1.0/NUM_CLASSES,
                'SER': neutral_probs, 'SER_emotion': 'Neutral', 'SER_confidence': 1.0/NUM_CLASSES,
                'META': neutral_probs, 'META_emotion': 'Neutral', 'META_confidence': 1.0/NUM_CLASSES,
                'inference_time': 0.001,
                'model_type': 'fallback',
                'model_status': {'ter': False, 'fer': False, 'ser': False, 'meta': False}
            }

def test_complete_system():
    """Test the complete meta-classifier system"""
    print("🧪 Testing Complete Meta-Classifier System...")
    
    # Initialize
    recognizer = CompleteMetaClassifier()
    
    # Test with sample data
    test_text = "I'm feeling absolutely fantastic and excited about this breakthrough!"
    test_result = recognizer.predict_multimodal(test_text)
    
    print("\n🎯 Test Results:")
    print(f"Meta-Classifier: {test_result['META_emotion']} ({test_result['META_confidence']:.2%})")
    print(f"TER: {test_result['TER_emotion']} ({test_result['TER_confidence']:.2%})")
    print(f"FER: {test_result['FER_emotion']} ({test_result['FER_confidence']:.2%})")
    print(f"SER: {test_result['SER_emotion']} ({test_result['SER_confidence']:.2%})")
    print(f"Model Type: {test_result['model_type']}")
    print(f"Inference Time: {test_result['inference_time']:.3f}s")
    print(f"Models Status: {test_result['model_status']}")
    
    return recognizer

if __name__ == "__main__":
    test_complete_system()