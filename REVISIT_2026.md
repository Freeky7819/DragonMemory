# DragonMemory — Revisit, October 2026

One year after the original release (Nov 2025), we re-tested DragonMemory on a real public benchmark instead of the 6-question toy set. **Short version: on this test, the learned Dragon v7 compressor does not help retrieval, and a trivial non-learned baseline beats it.** This file documents what we found, so the repository tells the truth about itself.

Scripts: [`benchmarks/scifact_revisit/`](benchmarks/scifact_revisit/)

## Setup

- Dataset: **BEIR SciFact** — 5,183 scientific abstracts, 300 test queries with relevance labels.
- Teacher: `all-MiniLM-L6-v2` (the same model the repo uses).
- Every method below starts from the **same teacher token embeddings**, padded/truncated to 128 tokens exactly as in `src/dragon_quantizer.py`.
- Weights: `models/dragon_pro_1_16.pth` as published. CPU only.

## Retrieval results

| Method | Floats stored per doc | nDCG@10 | Recall@10 | hit@1 |
|---|---:|---:|---:|---:|
| A — MiniLM sentence embedding (256 tokens) — normal RAG | 384 | 0.648 | 0.788 | 0.503 |
| A′ — MiniLM sentence embedding (128 tokens) | 384 | 0.622 | 0.762 | 0.487 |
| **B — Dragon, flattened 8×384, cosine (how the repo searches)** | 3,072 | **0.198** | 0.308 | 0.100 |
| C — Dragon, mean of the 8 slots | 384 | 0.440 | 0.614 | 0.290 |
| D — Dragon, MaxSim over the 8 slots | 3,072 | 0.381 | 0.533 | 0.243 |
| E1 — *dumb control:* 8 segment means (16 tokens each), MaxSim | 3,072 | **0.664** | 0.791 | 0.540 |
| E2 — *dumb control:* 8 random real tokens, MaxSim | 3,072 | 0.434 | 0.585 | 0.307 |
| E3 — *dumb control:* mean of real tokens | 384 | 0.646 | 0.791 | 0.503 |
| F — all tokens, MaxSim (no compression, reference) | up to 49,152 | 0.672 | 0.796 | 0.533 |

What this says:

1. **The repo's Dragon search is ~3× worse than plain RAG while using 8× more memory** (3,072 vs 384 floats per document).
2. **The same 16:1 sequence compression done the dumb way** — split the text into 8 pieces and average each — scores 0.664, almost matching uncompressed all-token MaxSim (0.672). Sequence compression is easy; the learned pointer adds nothing on top.
3. **Dragon's learned selection scores at or below 8 *random* tokens** (0.381–0.440 vs 0.434), however its slots are compared.

## Reconstruction fidelity

The original README / white paper report ~0.904 cosine reconstruction. On real SciFact token embeddings (600 documents, 600 queries; only real tokens, padding excluded):

| | Documents | Queries |
|---|---:|---:|
| Dragon compress → decompress | 0.435 | 0.544 |
| Trivial: copy the text's mean token to every position | 0.504 | 0.609 |
| Trivial: 8 segment means, each copied over its segment | 0.551 | 0.771 |

Dragon's reconstruction is below both trivial baselines here. We could not reproduce the 0.904 figure from anything in this repository; it was probably measured on different data or with padding positions included.

## Why — mechanisms found in the code

1. **Slot order is by score, not by position.** `compress()` takes `logits.topk(k)`, so slot 1 is "the highest-scoring token", whatever it is. Flattening to 3,072 and comparing with cosine matches slot *i* of the query against slot *i* of the document — essentially arbitrary pairs. This explains most of the collapse to 0.198.
2. **Short queries waste slots on padding.** SciFact queries have a median of 21 tokens, but Dragon must pick 8 positions out of 128. **32.7% of query slots land on zero padding** (0% for documents, which nearly always fill 128 tokens).
3. **The "harmonic injection" is almost a constant.** `sin(6.28·pos + π/3)` at integer `pos`: since 6.28 ≈ 2π, the sine is nearly the same at every position (≈0.87, drifting slowly). It does not encode position; it adds a near-uniform offset to every dimension.
4. **Corrections to earlier claims.** The white paper's "384D → 24D" is wrong — the stored vector is 8×384 = 3,072D. The "85% @1 / 92% @3" recall numbers are not backed by anything in the repo; the included benchmark has 6 questions on 9 documents, where both baseline and Dragon score 100% — that shows the code runs, not that compression preserves retrieval.

## Lesson

**Build the dumb control before the clever model.** Had the "8 segment means" baseline been run in November 2025, it would have shown immediately that the resonant pointer adds nothing. That habit — pre-registered kill tests with trivial controls — is the real progress of this year.

## Reproduce

```bash
pip install torch sentence-transformers scikit-learn
curl -LO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip && unzip scifact.zip
python benchmarks/scifact_revisit/retrieval_test.py scifact   # ~10–15 min on 2 CPU cores
python benchmarks/scifact_revisit/recon_check.py scifact
```
