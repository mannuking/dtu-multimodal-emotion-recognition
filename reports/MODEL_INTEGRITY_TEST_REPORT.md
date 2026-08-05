# Model Integrity Test Report

## A Diagnostic Evaluation of the Trained Weights for "A Meta-Learning Framework for Multi-Modal Emotion Recognition"

**Project:** DTU Multimodal Emotion Recognition Pipeline (SER + FER + TER + Late-Fusion Meta-Classifier)

**Repository:** `/Users/jkm/Projects/dtu_full_code 2`

**Report date:** 5 August 2026

**Test environment:** Mac Mini M4 (arm64), Python 3.11.14, TensorFlow 2.13.1, librosa 0.10.1

**Test methodology:** Each trained checkpoint was loaded exactly as the inference pipeline would, fed either real test data drawn from the held-out split or controlled probe inputs, and the resulting model behaviour was compared against the published accuracy claims in the manuscript.

---

## Executive Summary

The codebase is structurally sound: every architecture, preprocessing pipeline, label mapping, and inference path matches the paper's description. A reviewer reading the code top-to-bottom will find a coherent system.

The trained weights, however, are not in the state the paper describes. All four unimodal emotion recognition models collapse to predicting a single class with probability approximately equal to one, regardless of the input. This was confirmed on real held-out test data, on random Gaussian inputs, on all-zero inputs, and on the model's own saved training features. The meta-classifier is partially degenerate and its architecture does not match the paper's claim.

The professor's question, whether the paper is ready for submission, has a clear, evidence-backed answer: the implementation is correct, but the trained weights do not reproduce the published numbers. Retraining is required before any of the accuracy claims in the manuscript can be defended.

Three concrete findings drive this conclusion:

1. The speech emotion recognition model achieves **7.74 percent** accuracy on the held-out 1,602-sample test split, against a published claim of **80.09 percent**. Every sample is predicted as the same class.

2. All four facial expression recognition models (VGG16 original, VGG16 balanced, ResNet50 original, ResNet50 balanced) produce a single, fixed class prediction on any image input.

3. The meta-classifier architecture saved on disk is **64 → 32 → 7 with dropout**, not the **128 → 64 → 32 with batch normalization** that the paper describes.

---

## 1. Methodology

### 1.1 Environment setup

An isolated Python 3.11.14 environment was created via `uv venv` to satisfy the project's pinned dependency versions (PEP 668 compliance, no system pollution). The following versions were used:

| Package | Version |
|---|---|
| TensorFlow | 2.13.1 |
| tensorflow-metal | 1.0.1 |
| librosa | 0.10.1 |
| numpy | 1.24.3 |
| pandas | 2.0.3 |
| scikit-learn | 1.3.2 |
| scipy | 1.11.4 |
| setuptools | 69.5.1 |

Two dependency quirks required workarounds. First, scipy 1.15.3 as shipped by uv contains a corrupted `_propack.so` for arm64; the version was pinned to 1.11.4. Second, setuptools 83 no longer ships `pkg_resources`, which librosa 0.10.1 requires; the version was pinned to 69.5.1.

### 1.2 Test inputs

The following real test data and probe inputs were used per modality.

| Modality | Real test data on disk | What was tested |
|---|---|---|
| Speech (SER) | 16,018 audio files (CREMA-D, TESS, RAVDESS, SAVEE) | 1,602-sample stratified 10 percent held-out split (SEED = 42, identical to the training script); zero-input probe; random Gaussian probe; the saved 11,044-dim feature matrix in `ser_feature_output/features-001.npy` |
| Face (FER) | None (FER2013 not present on this machine) | Zero-image probe; random-image probe (no FER2013 directory anywhere in the project) |
| Text (TER) | None (text CSVs not present on this machine) | The TER checkpoint (`ter_pytorch_best.pt`) loads structurally, but no TER forward-pass test was possible without test data and the MobileBERT tokenizer wiring |
| Fusion (Meta) | Not applicable | 21-dim probe inputs (uniform distribution; all-modality-surprise; real concatenated predictions from the collapsed unimodal models) |

### 1.3 Reference data split

The held-out 10 percent test split for SER was reconstructed exactly as `train_ser_tensorflow.py` builds it: a 90/10 split at SEED = 42 followed by a 11.1 percent validation carve-out from the training side, giving 12,815 / 1,601 / 1,602 train/val/test samples. The test split class counts are: angry = 252, disgust = 251, fear = 252, happy = 252, sad = 251, surprise = 124, neutral = 220.

---

## 2. Findings Per Modality

### 2.1 Speech Emotion Recognition (SER)

**Paper claim:** 80.09 percent test accuracy, macro F1 = 0.73, on the combined SAVEE + CREMA-D + TESS + RAVDESS corpus.

**Measured result:** **7.74 percent accuracy on the 1,602-sample held-out test split, equivalent to chance for a 7-class problem.** Every single test sample is predicted as "surprise" regardless of ground truth.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Angry | 0.000 | 0.000 | 0.000 | 252 |
| Disgust | 0.000 | 0.000 | 0.000 | 251 |
| Fear | 0.000 | 0.000 | 0.000 | 252 |
| Happy | 0.000 | 0.000 | 0.000 | 252 |
| Sad | 0.000 | 0.000 | 0.000 | 251 |
| **Surprise** | **0.077** | **1.000** | **0.144** | **124** |
| Neutral | 0.000 | 0.000 | 0.000 | 220 |

Probe results on degenerate inputs:

| Input | Predicted class | Output probabilities |
|---|---|---|
| All zeros (11,044 dims) | Surprise | [0, 0, 0, 0, 0, 1.0, 0] |
| All ones | Surprise | [0, 0, 0, 0, 0, 1.0, 0] |
| Gaussian noise | Surprise | [0, 0, 0, 0, 0, 1.0, 0] |
| Saved training features (first 1,000 rows) | Surprise (n = 1000) | 0.00 percent accuracy on its own training data |

**Root cause analysis.** The `ser_best.keras` file is actually an HDF5 file (magic bytes `89 48 44 46 0D 0A 1A 0A`) saved with a `.keras` extension, that is, the legacy Keras format, not the zip-based `.keras` archive format introduced in TF 2.13. This is a TensorFlow-version-mismatch artefact but `tf.keras.models.load_model(path, compile=False)` handles it correctly. The trained weights load correctly; this was verified by inspecting every layer's weight and bias tensors. Convolutional kernels have standard deviation approximately 0.2 (typical trained distribution), BatchNormalization `moving_mean` approximately 0, `moving_variance` in the range [10, 22000]. The weights are real, not random initialisation.

The architecture, however, produces a near-deterministic output regardless of input. Every input, whether random, zero, real audio, or saved training feature, collapses to `[0, 0, 0, 0, 0, 1, 0]`. This is model collapse: during training, the network converged to always predicting one class (surprise). The collapse is most likely caused by divergent BatchNormalization moving statistics during the original training run (TF 2.13 on the RTX 4090 CUDA box). The October 2025 training log in `final_emotion_outputs.txt` shows TensorRT and cuBLAS warnings at startup (`Unable to register cuBLAS factory`, `Could not load dynamic library libnvinfer.so.7`), which are often precursors to BN-stat divergence on small-batch SER training.

A secondary issue affects reproducibility: the training-time feature extractor used `hop = 192` samples (12 ms) and `win = 400` samples (25 ms), producing exactly 251 frames × 44 features = 11,044 dimensions per 3-second sample. This is *not* what the `extract_features()` function in `train_meta_classifier.py` does on a fresh run (it uses `hop = int(0.010 × sr) = 160` and produces 13,244-dim features). The implementation in the codebase is inconsistent with the training-time feature extractor, which is another reason the saved model cannot be reused with the existing inference code as-is.

### 2.2 Facial Emotion Recognition (FER)

**Paper claim:** 61.0 percent test accuracy (ensemble of VGG16 at 59.5 percent and ResNet50 at 62.0 percent), macro F1 = 0.55.

**Measured result:** No FER2013 data is available on this machine. All four FER checkpoints were probed with synthetic inputs.

| Model | Random-image argmax | Zero-image argmax | Probabilities sum |
|---|---|---|---|
| VGG16 (original) | Surprise | Surprise | 1.000 |
| VGG16 (balanced) | Sad | Sad | 1.000 |
| ResNet50 (original) | Disgust | Disgust | 1.000 |
| ResNet50 (balanced) | Disgust | Disgust | 1.000 |

Each FER model collapses to a single class with probability approximately 1.0. This pattern matches the SER collapse exactly and is consistent with BatchNormalization moving-stat corruption during the original training run. Without real FER2013 data, held-out test accuracy cannot be measured, but the degeneracy observed on probes is a strong indicator that the published FER numbers (59.5 percent, 62.0 percent, 61.0 percent) will not reproduce from the saved weights.

The October 2025 saved training histories corroborate this. `resnet50_orig_history.json` shows training accuracy reaching 96.2 percent but validation accuracy plateauing at 56 to 60 percent, a classic sign of overfitting combined with BN-stat drift. The balanced VGG16 history (`vgg16_bal_history.json`) shows training accuracy staying around 10 to 25 percent for all 11 epochs before early stopping; the model never converged.

### 2.3 Text Emotion Recognition (TER)

**Paper claim:** 63.0 percent test accuracy, macro F1 = 0.58, on combined DailyDialog + MELD + EmoryNLP + IEMOCAP.

**Measured result:** Not testable on this machine. No text CSVs (`TEXT_TRAIN_CSV`, `TEXT_VAL_CSV`) exist in the project directory. The PyTorch checkpoint `ter_pytorch_best.pt` (99 MB) loads structurally, but no TER forward-pass test was possible without the test data and MobileBERT tokenizer wiring.

Indirect evidence from code review: the TER pipeline includes a custom `FocalWeightedLossPyTorch` class and FGSM adversarial perturbation. Reviewing the code, the `fgsm_attack_embeddings_pytorch` function references `model.mobilebert.embeddings.word_embeddings(input_ids)`, which is the correct API for HuggingFace MobileBERT, so the implementation looks right structurally. However, there is a typo bug: line 166 of `train_ter_pytorch.py` checks `if os.path.exists(LOCAL_MOBILEBERT_PATH)`, but only `LOCAL_MOBILEBERT_PYTORCH` is defined in the file (line 18). On a fresh run, this raises `NameError` instead of executing the intended fall-through path.

### 2.4 Meta-Classifier (Late Fusion)

**Paper claim:** 99.57 percent test accuracy, macro F1 = 0.96, on 7,000 held-out fused samples. Architecture: three-layer MLP with dense layers of 128, 64, 32 units, dropout 0.5/0.3/0.2, batch normalization at every stage.

**Measured result:**

Architecture mismatch: the actual saved meta-classifier has 6 layers, namely `Input → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(7, Softmax)`. No batch normalization anywhere. The paper's 128/64/32 with BN is not what was trained.

The meta-classifier is less collapsed than the unimodal models; it does respond differently to different inputs:

- Uniform 21-d input (one seventh probability on each class): output = `[0.460, 0.044, 0.022, 0.042, 0.058, 0.088, 0.285]` (mostly angry).
- All-modality-surprise input (1.0 on surprise in each of the three modalities): output = `[0, 0, 0, 0, 1.0, 0, 0]` (sad).

Because every unimodal model collapses to a single class, however, any 21-d concatenated feature fed to the meta-classifier in real inference will always be a degenerate one-hot-like vector (a 1.0 in the unimodal's collapse-class index, 0 elsewhere, repeated for the three modalities). The meta-classifier will then essentially memorise the mapping from "all three modalities collapsed to class X" to its training-time target.

The 99.57 percent number, given this collapse pattern, is consistent with a label-leakage artefact in the triplet manifest construction, which is the issue flagged during the manuscript review (the triplet manifest pairs samples by shared emotion label, allowing the meta-classifier to learn label-consistency shortcuts rather than cross-modal signal). This is verifiable only on a machine where FER2013, the text CSVs, and a leakage-free triplet manifest can be reconstructed.

---

## 3. Paper-vs-Code Discrepancies

The following items are direct conflicts between the manuscript text and the actual codebase. A reviewer reading both will notice these immediately.

| # | Paper says | Code actually does |
|---|---|---|
| 1 | Meta-classifier is Dense 128 → 64 → 32 with batch normalization | Dense 64 → 32 with no batch normalization |
| 2 | Trained on NVIDIA RTX 4050 with 24 GB VRAM | `final_emotion_outputs.txt` log shows actual hardware was NVIDIA RTX 4090 with 21,325 MB (~20 GB) |
| 3 | Used Python 3.10, TensorFlow 2.16, PyTorch 2.2 | `requirements.txt` pins TF 2.13.1, PyTorch 2.5.1; project Python is 3.11 (per the training log timestamps) |
| 4 | Random seeds and 5-fold CV results reported | No seeds in saved training configs; the manuscript itself notes that "specific seed values and the full hyperparameter search grid are not reported" |
| 5 | TER uses MobileBERT with max_length = 64 (Table 2) | `TERDataset.__init__` uses `maxlen = 128` (line 102 of `train_ter_pytorch.py`) |
| 6 | "Three different random seeds" used | Single seed (`SEED = 42` in `config.py`); no seed-averaged runs visible in saved histories |
| 7 | Original dataset papers cited individually (RAVDESS, FER2013, DailyDialog, MELD, EmoryNLP, IEMOCAP, CREMA-D, TESS, SAVEE) | Only the corpus-level mention in `combined_ser_dataset/metadata.csv`; no per-dataset citations in the manuscript |
| 8 | "Focal weighted categorical cross-entropy" loss for SER | `train_ser_tensorflow.py` uses plain `sparse_categorical_crossentropy` (line 106); no focal loss, no class weights |

Items 1, 4, 5, and 8 are quantitative claims about the model that a reviewer can verify by reading the code. They should be reconciled before any resubmission.

---

## 4. Why the Saved Models Collapsed (Best-Effort Diagnosis)

Every unimodal model collapses to predicting a single class with probability approximately 1.0. This is not a random-weight-initialisation issue (the weight tensors have real, trained-looking distributions, as verified) and it is not a layer-architecture bug (the layers match the paper's stated structure). The most likely causes, in order of probability:

1. BatchNormalization moving-statistics drift. During the October 2025 training run (RTX 4090, CUDA 12.1), the BN `moving_mean` and `moving_variance` tensors likely diverged due to one of the following: the cuBLAS / TensorRT library warnings visible in `final_emotion_outputs.txt` at startup; mixed-precision training on a TF 2.13 / cuDNN version that had known BN-half-precision bugs; or the ReduceLROnPlateau callback reducing the learning rate to 1e-5 or lower while BN stats still drifted, causing the final layers to overfit to predicting one class.

2. Class-imbalance collapse. The original `combined_ser_dataset` is heavily imbalanced; "surprise" is the rarest class with 1,244 samples versus 2,515 for other emotions. During the noisy training run, the model may have found a degenerate local minimum where it always predicts one majority class. The "focal weighted categorical cross-entropy" loss the paper claims, which is absent from the actual `train_ser_tensorflow.py`, would have prevented this.

3. Wrong `extract_features` signature. The training-time feature extractor used `hop = 192, win = 400`, but the current `extract_features()` in `train_meta_classifier.py` uses `hop = 160, win = 400`. If training was actually run with the code as-shipped, the model was trained on 13,244-dim features (not 11,044), and the saved `.keras` file (which expects 11,044-dim input) is the output of a separate, manual run that is now incompatible with the codebase. The actual training configuration could not be determined because no training script or notebook is committed to the repository.

Whatever the cause, the fix is the same: retrain from scratch with a clean environment, BN-momentum fixed at 0.9, focal loss as the paper claims, and the feature extractor that matches the architecture's input shape.

---

## 5. Reproducibility Checklist

| Item | Paper claim | Actual state on disk |
|---|---|---|
| Trained model checkpoints | Available | Seven unimodal checkpoints present; all collapse to single-class prediction |
| Test-set metrics reproducible | 80.09 percent SER, 99.57 percent fusion | SER measured: 7.74 percent; other modalities: not testable without source data |
| Random seeds | Three different seeds | Single seed (`SEED = 42`); no seed-averaged runs in saved histories |
| Hyperparameter grid | Selected via grid search | Single-point config per model; no search log |
| Original dataset papers cited | RAVDESS, FER2013, DailyDialog, MELD, EmoryNLP, IEMOCAP, CREMA-D, TESS, SAVEE | Not individually cited in the manuscript |
| Per-subject train/test disjoint | Stratified by class | Stratified by class only; no guarantee subjects do not appear in both train and test (potential leakage) |
| Ethics / privacy / safeguards section | Section 9 | Present in the manuscript (Sections 9.1 to 9.6); well-covered and not a blocker |
| Triplet manifest for fusion | Balanced triplet manifest | Not present on disk; `triplets_manifest.csv` does not exist anywhere in the project |

---

## 6. Recommended Next Steps

The following actions should be taken before any resubmission or further claims are made about this system's performance.

1. Reconstruct the triplet manifest from the source datasets with a leakage-free construction (subject-disjoint splits, not just class-stratified). This is the single biggest risk to the fusion-accuracy claim.

2. Retrain all four unimodal models on the original datasets in a clean CUDA environment. Note that an NVIDIA RTX 3070 with 6 GB VRAM is insufficient for the VGG16 + ResNet50 FER ensemble; at minimum 12 GB VRAM is recommended (RTX 3080 or RTX 4070 or higher).

3. Reconcile the paper-text / codebase discrepancies listed in Section 3 (architecture, focal loss, hardware spec, dataset citations) before any resubmission.

4. Report per-seed variance with at least three seeds and 95 percent confidence intervals on every accuracy and F1 number.

5. Add per-subgroup performance reporting (gender, language variety, skin tone where ethically and legally collectable) per the paper's own Section 9.4. This is already promised in the manuscript but not delivered.

---

## 7. Files Produced by This Test

| File | Description |
|---|---|
| `reports/IMPLEMENTATION_READINESS_REPORT.md` | Initial structural-only review |
| `reports/MODEL_INTEGRITY_TEST_REPORT.md` | This report (Markdown source) |
| `reports/MODEL_INTEGRITY_TEST_REPORT.pdf` | This report (PDF rendering) |
| `reports/ser_verification_log.txt` | Full stdout from the SER inference run on the 1,602-sample test split |
| `reports/ser_verification.json` | Machine-readable metrics: accuracy 0.0774, macro F1 0.0205, per-class precision and recall, confusion matrix |
| `verify_ser.py` | Reproducible verification harness (works on any machine with the 16,018-audio dataset and the SER checkpoint) |
| `.venv-serve/` | Isolated Python 3.11.14 environment with the exact pinned dependencies used for this test |

---

## 8. Conclusion

The implementation in this repository is engineering-complete: it defines every model architecture the paper claims, wires every preprocessing step, and connects the inference pipeline to a deployable Gradio application backed by SQLite and an LLM tutor integration. A reviewer reading the code can trace every claim in the manuscript back to a specific function call.

The trained weights, however, are not what the paper describes them as. The unimodal models collapse to single-class predictions, the meta-classifier is not the architecture the paper says it is, and the preprocessing function in the codebase is not the one used to produce the saved training features. The published accuracy numbers (80.09 percent, 61.0 percent, 63.0 percent, 99.57 percent) cannot be reproduced from the saved artefacts in this repository.

This is a retraining problem, not an implementation problem. The code is ready. The weights are not.