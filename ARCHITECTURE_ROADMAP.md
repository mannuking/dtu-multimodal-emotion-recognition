# DTU Multimodal Emotion Recognition — Architecture Roadmap to 82%

This document tracks the progression from the v2 baseline (68% test acc) toward the 82% target.

## Current baseline (v2 + 3-seed ensemble, 8 Aug 2026)

| Seed | Best val_acc | **Test acc** | **Test macro-F1** |
|------|-------------|-------------|-------------------|
| 42   | 0.7366      | 0.6490      | 0.6440            |
| 43   | 0.7362      | **0.6739**  | **0.6783**        |
| 44   | 0.7379      | 0.6350      | 0.6381            |
| **Ensemble** (avg softmax + 5-pass TTA each) | — | **0.70–0.74** (forecast) | ~0.70 |

Per-class F1 (best single seed, 43):  
- angry 0.67, disgust 0.86, fear 0.43, happy 0.63, neutral 0.64, sad 0.31, surprise 0.88

**Bottleneck classes:** fear (0.43) and sad (0.31). The wav2vec2-base model confuses fear with sad, and sad with neutral — well-known failure mode for base-size encoders on these paralinguistic distinctions.

**Honest ceiling for wav2vec2-base on this dataset: 72–75% test acc.**  
Adding more epochs or tweaking hyperparameters won't break through. Architectural change is required.

## Phase 1 — wav2vec2-large + SupCon (target: 76–82%)

**Implemented in `train_ser_v4.py`.**

| Change | Expected gain | Effort |
|--------|--------------|--------|
| wav2vec2-large (317M, 24 layers, 1024-dim) instead of -base (90M) | +5–8 pp | 1 day |
| Supervised Contrastive (SupCon) auxiliary loss on 128-dim projected embeddings (Khosla et al. NeurIPS 2020) | +2–4 pp | 4 hours |
| Bidirectional attention pooling head (forward + backward) instead of single attention | +1–2 pp | 2 hours |
| Gradient checkpointing on encoder (required to fit -large in 40GB A100) | — | 1 hour |
| 50 epochs cosine (large converges faster than base's 70) | — | config |

**Total expected lift: +8 to +14 pp.** Realistic test acc: **76–82%** single seed, **78–84%** ensemble.

**Cost:** ~5–6 hours per seed on A100, 3 seeds in parallel on dgxnp = ~6 hours wall-clock.

**To run:**
```bash
# Once: download wav2vec2-large to HF cache (~1.2 GB)
python3 -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large')"

# Then on HPC login:
cd ~/Research/dtu/dtu-multimodal-emotion-recognition
git pull origin main
bash scripts/train_ser_v4_ensemble.sbatch
# Watch: tail -f ser_v4.<JOBID>.out
```

## Phase 2 — Multimodal fusion (target: 80–88%)

**NOT YET IMPLEMENTED.** Planned.

| Modality | Pretrained encoder | Source | Status |
|----------|-------------------|--------|--------|
| Audio | wav2vec2-large (Phase 1 output) | ✓ ready | done in v4 |
| Text | MobileBERT (already trained, `ter_pytorch_best.pt` 95 MB) | ✓ ready | needs feature extractor wrapper |
| Visual | ResNet50/EfficientNet on FER2013 facial expressions | partial | train_meta_classifier_pytorch.py exists |

**Architecture (planned):**
```
audio_wav2vec2_large → (B, 1024, T) → StrongSERHead.v4 → (B, 256) ─┐
text_mobilebert     → (B, 768)        → Linear(768, 256)   → (B, 256) ─┼→ Concat → MLP → 7 classes
visual_resnet50     → (B, 2048)       → Linear(2048, 256)  → (B, 256) ─┘
```

**Cross-modal attention fusion** (rather than concat) is the published SOTA on IEMOCAP multimodal SER — adds another 1–2 pp over concat.

**Expected lift over Phase 1:** +4 to +8 pp. **Realistic ensemble test acc: 84–90%.**

**Effort:** ~1 week. The text encoder is ready, visual encoder needs a quick fine-tune on FER2013 + alignment with the audio manifest.

## Phase 3 — Data expansion (target: 88–92%)

**NOT YET STARTED.** Conditional on Phase 2 success.

Add to the combined SER dataset:
- **IEMOCAP** (5,000+ acted emotional utterances, 10 speakers) — adds ~3 pp
- **Emo-DB** (German emotional speech, 535 utterances) — adds ~1 pp
- **MELD** (multimodal EmotionLines, 13,000+ utterances from Friends TV) — adds ~3 pp
- Synthetic data via voice conversion (CycleGAN-VC or StarGAN-VC) — adds ~2 pp

Realistic ensemble test acc with expanded data: **88–92%**.

## Decision tree

```
After Phase 1 ensemble result lands:
├── Test acc ≥ 78%: Phase 1 succeeded. Proceed to Phase 2 (multimodal fusion).
├── Test acc 70–77%: Phase 1 partial success. Investigate bottlenecks
│   (per-class F1 on fear/sad). May need focal loss + class re-weighting.
└── Test acc < 70%: Phase 1 failed. STOP and investigate. Common causes:
    - wav2vec2-large not in HF cache (job fails immediately)
    - Gradient checkpointing not enabled (OOM)
    - SupCon weight too high (training collapses)
```

## Architecture decisions log

- **2026-08-08:** v2 ensemble baseline at 0.65–0.67 test acc. Bottleneck is wav2vec2-base's 768-dim representation, which can't capture the acoustic differences between fear and sad reliably.
- **2026-08-08:** Phase 1 design. wav2vec2-large + SupCon is the highest expected-lift / lowest-risk combination. HuBERT-large is an alternative but wav2vec2-large is more battle-tested on paralinguistic tasks.
- **SupCon rationale:** fear and sad are acoustically similar (both low arousal, low energy in some dimensions). SupCon explicitly pulls same-class samples together in embedding space, which should sharpen the boundary. Confirmed effective on ESC-50 and CREMA-D in published work.