"""
gpu_runtime.py — Single chokepoint for GPU/perf configuration.

Critical: GPU visibility MUST be forced visible to TF BEFORE the CUDA
runtime initializes. On some HPC stacks (DGX A100, CUDA 12 + driver 525+)
TF's import order matters — if any GPU library gets loaded before TF,
the GPU list comes back empty.

This module:
- Forces a CUDA pre-init via os.environ before importing TF.
- Enables mixed precision AFTER model construction (so pretrained weights
  aren't auto-cast incorrectly).
- Sets up MirroredStrategy.
- Provides deterministic seeding.
"""

import os
import random
import numpy as np


def _pre_tf_setup():
    """Force CUDA/TF init order before TF import."""
    # Use TF's built-in allocator (faster than the default on A100)
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    # Disable CUDA module lazy-loading so the GPU is fully enumerated
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    # Make TF see only GPUs visible per SLURM allocation
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
    # XLA flags for A100
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/opt/hpcx")


def enable_tf_perf(num_gpus: int | None = None, mixed_precision: bool = True):
    """Initialize TF with proper GPU visibility, mixed precision, XLA, strategy.

    Returns the strategy.

    Args:
        num_gpus: Hint for which strategy to pick. None = auto-detect.
        mixed_precision: If True, set mixed_float16 AFTER model is built
            (call enable_mixed_precision() inside strategy.scope()).
            If False, skip mixed precision entirely (for debug).
    """
    _pre_tf_setup()
    import tensorflow as tf

    # 1. GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
        # Surface info
        for g in gpus:
            print(f"  ✓ GPU device: {g.name}  type={g.device_type}")
    else:
        print("  ⚠️  no GPUs visible to TF — falling back to CPU")
        print("     (nvidia-smi may still show GPUs; CUDA driver / TF init mismatch)")

    # 2. XLA JIT (only useful with GPUs)
    try:
        tf.config.optimizer.set_jit(True)
        print("  ✓ XLA JIT enabled")
    except Exception as e:
        print(f"  ! XLA disabled: {e}")

    # 3. Resolve effective GPU count
    n = num_gpus if num_gpus is not None else len(gpus)
    n = min(n, len(gpus))
    print(f"  ✓ visible GPUs: {len(gpus)}, using: {n}")

    # 4. Pick strategy
    if n >= 2:
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        strategy = tf.distribute.MirroredStrategy()
    elif n == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.get_strategy()

    print(f"  ✓ strategy: {type(strategy).__name__} (workers={getattr(strategy, 'num_replicas_in_sync', 1)})")

    # 5. Mixed precision — DO NOT set globally. Pretrained models (VGG16,
    # ResNet50) have float32 weights and break with mixed_float16 globally.
    # Instead, callers should enable it INSIDE the model build block via
    # tf.keras.mixed_precision.set_global_policy('mixed_float16'), but only
    # AFTER model is instantiated and weights are loaded. For training from
    # scratch (SER 1D-CNN) this is safe; for transfer learning (FER) we
    # explicitly skip and use a manual cast.
    if mixed_precision and n >= 1:
        # Leave global policy as float32 by default. Call enable_mixed_precision()
        # explicitly when training from scratch.
        print("  ⓘ mixed precision deferred (caller enables inside model scope)")

    return strategy


def enable_mixed_precision():
    """Call this AFTER building a model from scratch (not transfer-learning)."""
    import tensorflow as tf
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"  ✓ mixed precision enabled: {tf.keras.mixed_precision.global_policy().name}")


def set_seed(seed: int) -> None:
    """Deterministic seeds for TF, NumPy, Python."""
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def global_batch_size(per_replica: int, strategy) -> int:
    """Scale per-replica batch to global batch size across all GPUs."""
    return per_replica * strategy.num_replicas_in_sync