# V8 Results — Audio+Text (V/A-Conditioned CMT + Dialog Context)

## Per-fold validation accuracy (5-fold LOSO, 3 seeds)

| Fold | Held-out | seed42 | seed43 | seed44 | Mean |
|------|----------|--------|--------|--------|------|
| 0 | Session1 | 0.6857 | 0.7161 | 0.7226 | 0.7081 |
| 1 | Session2 | 0.7713 | 0.7703 | 0.7546 | 0.7654 |
| 2 | Session3 | 0.6994 | 0.7124 | 0.7011 | 0.7043 |
| 3 | Session4 | 0.7507 | 0.7536 | 0.7352 | 0.7465 |
| 4 | Session5 | 0.7204 | 0.7252 | 0.7284 | 0.7247 |

**5-fold mean: 0.7298** (averaged across 15 runs)

## Comparison

| Stage | Accuracy |
|-------|----------|
| v5 audio-only ensemble (7-class) | 0.6836 |
| Audio-only ceiling (4-class) | 0.7092 |
| **v8 audio+text (LOSO)** | **0.7298** |

**+2.06 pp over audio-only ceiling, +4.62 pp over v5 ensemble.**

## Notes

- All numbers are val_acc. Held-out test set will be added in v9.
- Early stopping not enabled; fold-0 overfits after epoch 8.
- Architecture: V/A-conditioned cross-attention + dialog context transformer on frozen HuBERT+BERT.
