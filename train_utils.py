"""
train_utils.py — Shared training utilities for the DTU multimodal pipeline.
- Focal weighted categorical cross-entropy (paper Sec 5.1)
- Audio augmentation (noise, pitch shift, time shift, gain) — paper Sec 5.1
- Class weight computation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def class_weights(y: np.ndarray) -> dict:
    """Inverse-frequency class weights for imbalanced labels."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def focal_weighted_cce(alpha: np.ndarray, gamma: float = 2.0):
    """Focal weighted categorical cross-entropy (paper loss fn).
    alpha: per-class weight vector (numpy array, length num_classes).
    """
    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        # clip to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        # one-hot encode labels
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(alpha)[0])
        # cross-entropy
        ce = -y_true_oh * tf.math.log(y_pred)
        # focal modulator
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        focal = tf.pow(1.0 - p_t, gamma)
        # weighted
        weighted = alpha * y_true_oh * ce
        return tf.reduce_mean(tf.reduce_sum(focal[..., None] * weighted, axis=-1))

    return loss


# ---- Audio augmentation (paper Sec 5.1) ----

def audio_augment(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Apply one of: noise injection, pitch shift, time shift, gain variation.

    y: 1D numpy array of audio samples.
    sr: sample rate.
    rng: numpy Generator (np.random.default_rng(seed)) for reproducibility.

    Returns augmented audio. Designed for short emotion clips (~3s).
    """
    choice = rng.integers(0, 4)
    if choice == 0:
        # noise injection
        noise = rng.normal(0, 0.005, size=y.shape).astype(y.dtype)
        return (y + noise).astype(y.dtype)
    elif choice == 1:
        # pitch shift via resampling trick (simple version, no librosa dependency)
        factor = float(rng.uniform(0.9, 1.1))
        new_len = max(1, int(len(y) / factor))
        idx = (np.arange(new_len) * factor).astype(int)
        idx = idx[idx < len(y)]
        return y[idx]
    elif choice == 2:
        # time shift (roll)
        shift = int(rng.integers(-int(0.1 * sr), int(0.1 * sr)))
        return np.roll(y, shift)
    else:
        # gain variation
        gain = float(rng.uniform(0.8, 1.2))
        return (y * gain).astype(y.dtype)


def make_keras_audio_augment(sr: int):
    """Returns a Keras preprocessing function that augments audio on the fly.

    Use with `tf.data.Dataset.map(...)` on raw audio batches. To stay
    fast, augmentation is light: noise + gain only (paper lists 4
    techniques but spec doesn't say all 4 are applied every batch).
    """
    def augment(wav):
        # wav is (samples,) float32
        # random noise
        noise = tf.random.normal(tf.shape(wav), stddev=0.005, dtype=wav.dtype)
        # random gain
        gain = tf.random.uniform([], 0.85, 1.15, dtype=wav.dtype)
        return wav * gain + noise

    return augment