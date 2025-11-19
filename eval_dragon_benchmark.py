#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate DragonMemory vs baseline embedding on mini RAG benchmark.

Baseline:
    - SentenceTransformer('all-MiniLM-L6-v2'), sentence_embedding

Dragon:
    - same teacher model, token_embeddings -> DragonNLP.compress (1:16) -> flatten (8x384 = 3072)

Measures:
    - retrieval hit@1, hit@3, MRR@3 for baseline and Dragon
    - basic info about dimensions and "sequence compression" (128 -> 8 positions)
"""

import os
import sys
import json
import math
import argparse
from typing import List, Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Ensure we can import memory_v3_model.py from src/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from memory_v3_model import DragonNLP  # type: ignore


def load_docs(path: str) -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def load_qa(path: str) -> List[Dict]:
    qa = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qa.append(json.loads(line))
    return qa


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (D,) or (N,D)
    b: (M,D)
    returns:
        if a is (D,) -> (M,)
        if a is (N,D) -> (N,M)
    """
    if a.ndim == 1:
        a = a[None, :]
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def build_baseline_index(
    docs: List[Dict],
    model: SentenceTransformer,
    device: torch.device,
) -> (np.ndarray, List[str]):
    """Sentence embedding for each document."""
    texts = [d["text"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]

    # Sentence embedding (standard RAG baseline)
    emb = model.encode(
        texts,
        output_value="sentence_embedding",
        convert_to_tensor=True,
        device=str(device),
        show_progress_bar=True,
    )
    emb = emb.cpu().numpy()  # (N, d)
    return emb, doc_ids


def encode_tokens(
    model: SentenceTransformer,
    text: str,
    device: torch.device,
    d_model: int,
) -> torch.Tensor:
    """
    Returns token_embeddings for given text (T, d_model).
    """
    with torch.no_grad():
        token_emb = model.encode(
            text,
            output_value="token_embeddings",
            convert_to_tensor=True,
            device=str(device),
        )
    # SentenceTransformers sometimes returns a list if input is a list of strings
    if isinstance(token_emb, list):
        if len(token_emb) == 0:
            return torch.zeros(1, d_model, device=device)
        token_emb = token_emb[0]
    if token_emb.ndim == 1:
        token_emb = token_emb.unsqueeze(0)
    return token_emb  # (T, d_model)


def build_dragon_index(
    docs: List[Dict],
    teacher: SentenceTransformer,
    dragon: DragonNLP,
    device: torch.device,
) -> (np.ndarray, List[str]):
    """
    For each document:
        teacher token_embeddings -> pad/truncate to seq_len -> Dragon.compress -> flatten (K * d_model)
    """
    d_model = dragon.d_model
    seq_len = dragon.seq_len

    vectors = []
    doc_ids = []

    dragon.eval()

    for d in docs:
        text = d["text"]
        token_emb = encode_tokens(teacher, text, device, d_model)  # (T, d_model)

        # Pad / truncate na seq_len
        T = token_emb.shape[0]
        padded = torch.zeros(1, seq_len, d_model, device=device)
        slen = min(T, seq_len)
        padded[0, :slen, :] = token_emb[:slen, :]

        with torch.no_grad():
            out = dragon.compress(padded)
            if isinstance(out, (tuple, list)):
                compressed = out[0]          # take only latent
            else:
                compressed = out

        vec = compressed.view(-1).cpu().numpy()  # (K * d_model,)
        vectors.append(vec)
        doc_ids.append(d["doc_id"])

    return np.stack(vectors, axis=0), doc_ids  # (N, K*d_model)


def eval_retrieval(
    qa: List[Dict],
    doc_ids: List[str],
    index_vectors: np.ndarray,
    embed_fn,
    k: int = 3,
) -> Dict[str, float]:
    """
    embed_fn(question: str) -> np.ndarray  (vector for query)
    index_vectors: (N, D)
    doc_ids: list of doc_id (len N)

    Returns hit@1, hit@3, MRR@3.
    """
    hits1 = 0
    hits3 = 0
    mrr3_sum = 0.0
    n = 0

    doc_id_array = np.array(doc_ids)

    for item in qa:
        q = item["question"]
        gold_docs = set(item["doc_ids"])

        q_vec = embed_fn(q)  # (D,)
        sims = cosine_sim(q_vec, index_vectors)[0]  # (N,)
        # Sort desc
        order = np.argsort(-sims)

        top_k = order[:k]
        top_k_ids = doc_id_array[top_k]

        # hit@1
        if top_k_ids[0] in gold_docs:
            hits1 += 1

        # hit@3
        if any(did in gold_docs for did in top_k_ids):
            hits3 += 1

        # MRR@3
        rank = None
        for i_idx, idx in enumerate(top_k):
            if doc_ids[idx] in gold_docs:
                rank = i_idx + 1  # 1-based
                break
        if rank is not None:
            mrr3_sum += 1.0 / rank

        n += 1

    return {
        "hit@1": hits1 / max(1, n),
        "hit@3": hits3 / max(1, n),
        "mrr@3": mrr3_sum / max(1, n),
        "n_questions": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Dragon vs baseline RAG benchmark (toy dataset).")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join("benchmarks", "toy_rag"),
        help="Directory with docs.jsonl and qa.jsonl",
    )
    parser.add_argument(
        "--dragon",
        type=str,
        default="dragon_pro_1_16.pth",
        help="Path to DragonNLP checkpoint (.pth).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Top-k for retrieval metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda | cpu | auto",
    )
    args = parser.parse_args()

    docs_path = os.path.join(args.dataset_dir, "docs.jsonl")
    qa_path = os.path.join(args.dataset_dir, "qa.jsonl")

    if not os.path.exists(docs_path) or not os.path.exists(qa_path):
        print(f"[ERROR] Cannot find docs/qa in {args.dataset_dir}")
        print(f"  expected: {docs_path}")
        print(f"            {qa_path}")
        sys.exit(1)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading dataset...")
    docs = load_docs(docs_path)
    qa = load_qa(qa_path)
    print(f"[INFO] Number of documents: {len(docs)}, number of questions: {len(qa)}")

    # Teacher model
    print("[INFO] Loading teacher model (all-MiniLM-L6-v2)...")
    teacher = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

    # Baseline index
    print("[INFO] Building BASELINE index...")
    baseline_index, baseline_doc_ids = build_baseline_index(docs, teacher, device)
    baseline_dim = baseline_index.shape[1]

    # Dragon model
    print("[INFO] Loading DragonNLP...")
    dragon = DragonNLP(d_model=384, seq_len=128, ratio=16).to(device)
    state = torch.load(args.dragon, map_location=device)
    dragon.load_state_dict(state)
    dragon.eval()

    # Dragon index
    print("[INFO] Building DRAGON index (1:16 compression)...")
    dragon_index, dragon_doc_ids = build_dragon_index(docs, teacher, dragon, device)
    dragon_dim = dragon_index.shape[1]

    # Verify that we have the same doc_ids
    assert baseline_doc_ids == dragon_doc_ids, "Doc IDs in baseline and Dragon index do not match."

    # Embed functions for questions
    def embed_q_baseline(text: str) -> np.ndarray:
        v = teacher.encode(
            text,
            output_value="sentence_embedding",
            convert_to_tensor=True,
            device=str(device),
        )
        return v.cpu().numpy()

    def embed_q_dragon(text: str) -> np.ndarray:
        token_emb = encode_tokens(teacher, text, device, dragon.d_model)
        T = token_emb.shape[0]
        padded = torch.zeros(1, dragon.seq_len, dragon.d_model, device=device)
        slen = min(T, dragon.seq_len)
        padded[0, :slen, :] = token_emb[:slen, :]
        with torch.no_grad():
            out = dragon.compress(padded)
            if isinstance(out, (tuple, list)):
                compressed = out[0]
            else:
                compressed = out
        return compressed.view(-1).cpu().numpy()

    print("[INFO] Eval BASELINE...")
    baseline_metrics = eval_retrieval(
        qa,
        baseline_doc_ids,
        baseline_index,
        embed_q_baseline,
        k=args.k,
    )

    print("[INFO] Eval DRAGON...")
    dragon_metrics = eval_retrieval(
        qa,
        dragon_doc_ids,
        dragon_index,
        embed_q_dragon,
        k=args.k,
    )

    # Calculate "sequence compression"
    seq_len = dragon.seq_len
    k_positions = seq_len // 16  # ratio=16
    seq_compression = seq_len / k_positions if k_positions > 0 else None

    print("\n================= RESULTS =================")
    print(f"Number of questions: {baseline_metrics['n_questions']}")
    print(f"Baseline dim: {baseline_dim}")
    print(f"Dragon dim:   {dragon_dim}")
    if seq_compression is not None:
        print(f"Sequence compression (T -> K): {seq_len} -> {k_positions} (~1:{int(seq_compression)})")
    print("--------------------------------------------")
    print("BASELINE:")
    print(f"  hit@1 = {baseline_metrics['hit@1']:.3f}")
    print(f"  hit@3 = {baseline_metrics['hit@3']:.3f}")
    print(f"  mrr@3 = {baseline_metrics['mrr@3']:.3f}")
    print("DRAGON:")
    print(f"  hit@1 = {dragon_metrics['hit@1']:.3f}")
    print(f"  hit@3 = {dragon_metrics['hit@3']:.3f}")
    print(f"  mrr@3 = {dragon_metrics['mrr@3']:.3f}")
    print("=============================================\n")

    # Optional: calculate "index size" in MB (float32)
    n_docs = len(docs)
    baseline_bytes = n_docs * baseline_dim * 4
    dragon_bytes = n_docs * dragon_dim * 4
    print(f"[INFO] Estimated index size (float32):")
    print(f"  baseline ≈ {baseline_bytes / (1024*1024):.4f} MB")
    print(f"  dragon   ≈ {dragon_bytes / (1024*1024):.4f} MB")
    print(" (Dragon currently uses a larger latent vector – here we mainly measure retrieval quality;")
    print("  later we can add float16/int8/PQ for better index compression.)")

    # --------------------------------------------
    #  ADDITIONAL: embedding compression (teacher tokens -> Dragon latents)
    # --------------------------------------------
    seq_len = dragon.seq_len
    k_positions = dragon.k
    d_model = dragon.d_model

    # Hypothetical scenario:
    # - baseline: store all teacher token embeddings (seq_len * d_model)
    # - dragon: store only k latents (k * d_model)
    n_docs = len(docs)
    teacher_token_bytes = n_docs * seq_len * d_model * 4
    dragon_latent_bytes = n_docs * k_positions * d_model * 4

    print("\n[INFO] Hypothetical embedding storage (teacher tokens vs Dragon latents, float32):")
    print(f"  teacher tokens: {teacher_token_bytes / (1024*1024):.4f} MB "
          f" (seq_len={seq_len}, d_model={d_model})")
    print(f"  dragon latents: {dragon_latent_bytes / (1024*1024):.4f} MB "
          f" (k={k_positions}, d_model={d_model})")
    if k_positions > 0:
        print(f"  token compression factor ≈ {seq_len / k_positions:.1f}x")
    print("  (This is the true 1:{:.0f} compression per sequence: {} -> {} positions.)"
          .format(seq_len / max(1, k_positions), seq_len, k_positions))


if __name__ == "__main__":
    main()
