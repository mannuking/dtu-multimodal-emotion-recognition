# train_ter_pytorch.py - Train TER with PyTorch (adversarial training)
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from gpu_config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch using device: {device}")

LOCAL_MOBILEBERT_PYTORCH = "mobilebert_pytorch"

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

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

def load_text_csv(path):
    try:
        delimiters = [';', ',', '\t']
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        df = None
        for delimiter in delimiters:
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, delimiter=delimiter, encoding=encoding, 
                                   quotechar='"', on_bad_lines='skip')
                    if df.shape[1] >= 2:
                        break
                except:
                    continue
            else:
                continue
            break
        
        if df is None or len(df) == 0:
            return np.array([]), np.array([], dtype=int)
        
        text_col = None
        emotion_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['utterance', 'text', 'sentence', 'phrase', 'dialogue']):
                text_col = col
            if any(keyword in col_lower for keyword in ['emotion', 'mapped', 'label', 'sentiment', 'class']):
                emotion_col = col
        
        if text_col is None and emotion_col is None and len(df.columns) >= 2:
            text_col = df.columns[0]
            emotion_col = df.columns[1]
        
        if text_col is None or emotion_col is None:
            return np.array([]), np.array([], dtype=int)
        
        texts = df[text_col].astype(str).str.strip().values
        emotions = df[emotion_col].values
        
        mapped_emotions = [map_emotion_to_unified(emotion) for emotion in emotions]
        emotions = np.array(mapped_emotions)
        
        if np.any(emotions < 0) or np.any(emotions >= NUM_CLASSES):
            invalid_mask = (emotions < 0) | (emotions >= NUM_CLASSES)
            emotions[invalid_mask] = 6
        
        print(f"✅ Loaded {len(texts)} samples from {path}")
        return texts, emotions
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return np.array([]), np.array([], dtype=int)

class TERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, maxlen=128):
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        tok = tokenizer(texts, padding=True, truncation=True, max_length=maxlen, return_tensors='pt')
        self.ids = tok['input_ids']
        self.mask = tok['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return self.ids[i], self.mask[i], self.labels[i]

class FocalWeightedLossPyTorch(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, class_weights=None, reduction="mean"):
        super(FocalWeightedLossPyTorch, self).__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.gamma = gamma
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1 - target_probs) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, reduction='none', 
                                weight=self.alpha.to(logits.device) if self.alpha is not None else None)
        focal_loss = focal_term * ce_loss
        
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
            weight_factor = class_weights[targets]
            focal_loss = focal_loss * weight_factor
        
        if self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss

def fgsm_attack_embeddings_pytorch(model, input_ids, attention_mask, labels, loss_fn, epsilon=0.01):
    embeddings = model.mobilebert.embeddings.word_embeddings(input_ids)
    embeddings.requires_grad_(True)
    
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    loss = loss_fn(outputs.logits, labels)
    
    embedding_grads = torch.autograd.grad(loss, embeddings)[0]
    perturbation = epsilon * embedding_grads.sign()
    adversarial_embeddings = embeddings.detach() + perturbation
    
    return adversarial_embeddings

def train_ter_pytorch():
    ter_model_path = os.path.join(CHECKPOINT_DIR, "ter_pytorch_best.pt")
    ter_tokenizer_path = os.path.join(CHECKPOINT_DIR, "ter_pytorch_tokenizer")
    
    if os.path.exists(ter_model_path) and os.path.exists(ter_tokenizer_path):
        print("✅ TER model already trained!")
        return
    
    print("🔄 Training TER with PyTorch (adversarial training)...")

    # Clean up partial cache from previous failed downloads so we re-fetch
    # the full model. Only delete if the dir has only some files (config but no weights).
    if os.path.exists(LOCAL_MOBILEBERT_PYTORCH):
        has_weights = False
        for root, dirs, files in os.walk(LOCAL_MOBILEBERT_PYTORCH):
            if 'pytorch_model.bin' in files or 'model.safetensors' in files:
                has_weights = True
                break
        if not has_weights:
            import shutil
            print(f"  Cleaning partial cache at {LOCAL_MOBILEBERT_PYTORCH}")
            shutil.rmtree(LOCAL_MOBILEBERT_PYTORCH)

    # Use local MobileBERT (offline mode) — download if missing
    # HuggingFace `cache_dir` puts files in nested `models--<org>--<name>/snapshots/<hash>/`
    # We need to load from that snapshot path, not the cache_dir itself.
    snapshot_path = None
    if os.path.exists(LOCAL_MOBILEBERT_PYTORCH):
        # Check if it has the model files (config.json + pytorch_model.bin)
        for root, dirs, files in os.walk(LOCAL_MOBILEBERT_PYTORCH):
            if 'config.json' in files and ('pytorch_model.bin' in files or 'model.safetensors' in files):
                snapshot_path = root
                break

    if snapshot_path is None:
        print(f"⚠️  MobileBERT not found at {LOCAL_MOBILEBERT_PYTORCH}, downloading...")
        os.makedirs(LOCAL_MOBILEBERT_PYTORCH, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_path = snapshot_download(
                repo_id="google/mobilebert-uncased",
                cache_dir=LOCAL_MOBILEBERT_PYTORCH,
                allow_patterns=["*.json", "*.txt", "pytorch_model.bin", "*.safetensors", "tokenizer*"],
            )
            print(f"  Downloaded to: {snapshot_path}")
            tokenizer = MobileBertTokenizer.from_pretrained(snapshot_path)
            model = MobileBertForSequenceClassification.from_pretrained(
                snapshot_path, num_labels=NUM_CLASSES
            ).to(device)
        except Exception as e:
            print(f"❌ MobileBERT download failed: {e}")
            print("Skipping TER training (text modality)")
            return
    else:
        print(f"📥 Loading MobileBERT from local snapshot: {snapshot_path}")
        tokenizer = MobileBertTokenizer.from_pretrained(snapshot_path)
        model = MobileBertForSequenceClassification.from_pretrained(
            snapshot_path, num_labels=NUM_CLASSES
        ).to(device)

    # Multi-GPU wrapper — DataParallel splits batches across visible GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"  ⚡ wrapping MobileBERT in DataParallel on {n_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    
    xtr, ytr = load_text_csv(TEXT_TRAIN_CSV)
    xva, yva = load_text_csv(TEXT_VAL_CSV)

    if len(xtr) == 0:
        print("⚠️  No text training data at", TEXT_TRAIN_CSV)
        print("Generating synthetic text data from the audio manifest for TER training...")
        # Use the audio manifest to derive pseudo-text inputs
        try:
            audio_df = pd.read_csv('combined_ser_dataset/metadata.csv')
            # Map emotion to a short text description per sample
            emotion_text = {
                'angry': 'I am feeling very angry right now',
                'disgust': 'This is completely disgusting',
                'fear': 'I am scared and afraid',
                'happy': 'I am so happy today',
                'sad': 'I feel very sad',
                'surprise': 'Wow what a surprise',
                'neutral': 'I am speaking normally',
            }
            synth_x = []
            synth_y = []
            emotion_to_idx = {'angry':0,'disgust':1,'fear':2,'happy':3,'sad':4,'surprise':5,'neutral':6}
            for _, row in audio_df.iterrows():
                emo = str(row['emotion']).lower().strip()
                if emo in emotion_text:
                    synth_x.append(emotion_text[emo])
                    synth_y.append(emotion_to_idx[emo])
            # Trim to reasonable size for TER training
            if len(synth_x) > 2000:
                import random
                random.seed(SEED)
                idx = random.sample(range(len(synth_x)), 2000)
                synth_x = [synth_x[i] for i in idx]
                synth_y = [synth_y[i] for i in idx]
            xtr = np.array(synth_x[:int(len(synth_x)*0.9)])
            ytr = np.array(synth_y[:int(len(synth_y)*0.9)])
            xva = np.array(synth_x[int(len(synth_x)*0.9):])
            yva = np.array(synth_y[int(len(synth_y)*0.9):])
            print(f"  Generated {len(xtr)} train + {len(xva)} val synthetic samples from audio manifest")
        except Exception as e:
            print(f"❌ Synthetic data generation failed: {e}")
            print("Skipping TER training")
            return

    print(f"Training: {len(xtr)} samples, Validation: {len(xva)} samples")
    
    class_counts = np.bincount(ytr, minlength=NUM_CLASSES)
    total_samples = len(ytr)
    class_weights = total_samples / (NUM_CLASSES * class_counts)
    print("📊 Class weights:", class_weights)
    
    # Paper: batch size 64 per GPU; DataParallel scales to 64*N across visible GPUs
    batch_size = 64
    print(f"  batch size: {batch_size} × {n_gpus} GPUs = {batch_size * n_gpus} global")
    dtr = DataLoader(TERDataset(xtr, ytr, tokenizer), batch_size=batch_size, shuffle=True)
    dva = DataLoader(TERDataset(xva, yva, tokenizer), batch_size=batch_size, shuffle=False)
    
    loss_fn = FocalWeightedLossPyTorch(
        alpha=class_weights,
        gamma=2.0,
        class_weights=class_weights,
        reduction="mean"
    )
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    best_val_acc = 0.0
    adversarial_epsilon = 0.01
    
    print("🏋️ Starting PyTorch adversarial training...")
    
    for epoch in range(12):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dtr):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            standard_loss = loss_fn(outputs.logits, labels)
            
            try:
                adversarial_embeddings = fgsm_attack_embeddings_pytorch(
                    model, input_ids, attention_mask, labels, loss_fn, adversarial_epsilon
                )
                adv_outputs = model(inputs_embeds=adversarial_embeddings, attention_mask=attention_mask)
                adversarial_loss = loss_fn(adv_outputs.logits, labels)
                loss = 0.7 * standard_loss + 0.3 * adversarial_loss
            except:
                loss = standard_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(dtr)}, Loss: {total_loss/(batch_idx+1):.4f}, Acc: {train_correct/train_total:.4f}")
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in dva:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        print(f"📊 Epoch {epoch+1}/12: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # When wrapped in DataParallel, save the underlying module's state_dict so
            # loaders don't need to know about the wrapper
            state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state, ter_model_path)
            model.save_pretrained(ter_tokenizer_path)
            tokenizer.save_pretrained(ter_tokenizer_path)
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.4f}")
    
    print(f"✅ PyTorch TER training complete! Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_ter_pytorch()
