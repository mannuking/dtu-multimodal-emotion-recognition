"""
gpu_runtime.py — Single chokepoint for GPU/perf configuration.

Read these from any training script and you automatically get:
- TF mixed precision (mixed_float16) where supported
- XLA JIT compilation for the major TF ops
- tf.distribute.MirroredStrategy over all visible GPUs (no-op on 1 GPU)
- A deterministic seeding pass for reproducibility

Designed for PARAM Siddhi-AI (1, 2, 4 or 8 A100-SXM4-40GB visible per node).
A100 + CUDA 12 supports mixed_float16 / bfloat16; A100 also benefits from XLA.
"""

import os
import random
import numpy as np


def enable_tf_perf(num_gpus: int | None = None) -> "tf.distribute.Strategy":
    """Enable mixed precision + XLA + MirroredStrategy. Returns the strategy.

    Call this BEFORE importing/building any model. Returns either a
    MirroredStrategy (≥1 GPU detected) or a OneDeviceStrategy (CPU fallback).
    """
    import tensorflow as tf

    # 1. Mixed precision — A100 supports fp16 natively. Note: keep loss
    # in fp32 (handled by LossScaleOptimizer wrapper inside Keras). The
    # global policy converts weights + activations to fp16.
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"  ✓ mixed precision: {tf.keras.mixed_precision.global_policy().name}")
    except Exception as e:
        print(f"  ! mixed precision disabled: {e}")

    # 2. XLA JIT — A100-specific graph compilation. Falls back gracefully if
    # an op doesn't support XLA yet.
    try:
        tf.config.optimizer.set_jit(True)
        print("  ✓ XLA JIT enabled")
    except Exception as e:
        print(f"  ! XLA disabled: {e}")

    # 3. Allow GPU memory growth (so multiple processes don't fight for VRAM)
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    # 4. Determine effective GPU count
    n = num_gpus if num_gpus is not None else len(gpus)
    n = min(n, len(gpus))
    print(f"  ✓ visible GPUs: {len(gpus)}, using: {n}")

    # 5. Choose strategy
    if n >= 2:
        # Use cross-device communication via NCCL (default for GPU mirrors)
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
        strategy = tf.distribute.MirroredStrategy()
    elif n == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.get_strategy()  # CPU

    print(f"  ✓ strategy: {type(strategy).__name__} (workers={getattr(strategy, 'num_replicas_in_sync', 1)})")
    return strategy


def set_seed(seed: int) -> None:
    """Deterministic seeds for TF, NumPy, Python."""
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # TF deterministic ops
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def global_batch_size(per_replica: int, strategy) -> int:
    """Scale per-replica batch to global batch size across all GPUs."""
    return per_replica * strategy.num_replicas_in_sync