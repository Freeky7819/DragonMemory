"""DragonMemory revisit (Oct 2026): fair retrieval test on BEIR SciFact.

Compares, on the same teacher (all-MiniLM-L6-v2) token embeddings:
  A   MiniLM sentence embedding (what normal RAG stores)        384 floats/doc
  B   Dragon, flattened 8x384, cosine (as in the repo)          3072
  C   Dragon, mean of the 8 slots                               384
  D   Dragon, MaxSim late interaction over the 8 slots          3072
  E1  Dumb control: 8 segment means (16 tokens each), MaxSim     3072
  E2  Dumb control: 8 random real tokens, MaxSim                 3072
  E3  Dumb control: mean of all real tokens                      384
  F   All real tokens, MaxSim (ColBERT-style, no compression)    up to 128x384
Reconstruction fidelity is checked separately in recon_check.py.
Usage: python retrieval_test.py path/to/scifact
"""
import json, sys, time, math, random
import numpy as np, torch, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = sys.argv[1] if len(sys.argv) > 1 else "scifact"  # folder from https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
sys.path.insert(0, REPO)
from src.memory_v3_model import DragonNLP

torch.manual_seed(0); random.seed(0); np.random.seed(0)
torch.set_num_threads(2)
L, D, K = 128, 384, 8

corpus = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl")]
queries = {q["_id"]: q["text"] for q in map(json.loads, open(f"{DATA}/queries.jsonl"))}
qrels = {}
for i, line in enumerate(open(f"{DATA}/qrels/test.tsv")):
    if i == 0: continue
    qid, did, s = line.strip().split("\t")
    if int(s) > 0: qrels.setdefault(qid, set()).add(did)
qids = sorted(qrels)
doc_ids = [d["_id"] for d in corpus]
doc_txt = [(d.get("title", "") + ". " + d["text"]).strip() for d in corpus]
q_txt = [queries[q] for q in qids]
print(f"docs={len(doc_ids)} test queries={len(qids)}", flush=True)

teacher = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
dragon = DragonNLP(d_model=D, seq_len=L, ratio=16)
dragon.load_state_dict(torch.load(f"{REPO}/models/dragon_pro_1_16.pth", map_location="cpu"))
dragon.eval()

def tokens(texts):
    """Teacher token embeddings, padded/truncated to 128 exactly like the repo."""
    out = teacher.encode(texts, output_value="token_embeddings", batch_size=64,
                         convert_to_tensor=False, show_progress_bar=False)
    X = torch.zeros(len(texts), L, D); n = torch.zeros(len(texts), dtype=torch.long)
    for i, t in enumerate(out):
        s = min(t.shape[0], L); X[i, :s] = t[:s]; n[i] = s
    return X, n

def sent(texts, max_len):
    teacher.max_seq_length = max_len
    e = teacher.encode(texts, batch_size=64, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False)
    teacher.max_seq_length = 256
    return e

t0 = time.time()
Xd, nd = tokens(doc_txt); Xq, nq = tokens(q_txt)
print(f"token embeddings {time.time()-t0:.0f}s; doc tokens median={nd.median().item()} "
      f"(>=128: {(nd>=L).float().mean():.0%}); query tokens median={nq.median().item()}", flush=True)

@torch.no_grad()
def drag(X, bs=128):
    C, P = [], []
    for i in range(0, len(X), bs):
        c, p, _ = dragon.compress(X[i:i+bs]); C.append(c); P.append(p)
    return torch.cat(C), torch.cat(P)

Cd, Pd = drag(Xd); Cq, Pq = drag(Xq)
# how often does Dragon pick padding positions?
pos_d = (Pd * L).round().long(); pos_q = (Pq * L).round().long()
pad_d = (pos_d >= nd[:, None]).float().mean().item()
pad_q = (pos_q >= nq[:, None]).float().mean().item()
print(f"Dragon slots on padding: docs {pad_d:.1%}, queries {pad_q:.1%}", flush=True)

def mask(n):
    return (torch.arange(L)[None, :] < n[:, None])

def seg_means(X, n):
    out = torch.zeros(len(X), K, D)
    for i in range(len(X)):
        r = X[i, :n[i]]; chunks = torch.tensor_split(r, min(K, len(r)))
        for j, c in enumerate(chunks): out[i, j] = c.mean(0)
    return out

def rand_tok(X, n):
    out = torch.zeros(len(X), K, D)
    for i in range(len(X)):
        idx = torch.randperm(int(n[i]))[:K]; out[i, :len(idx)] = X[i, idx]
    return out

def mean_tok(X, n):
    m = mask(n).float().unsqueeze(-1)
    return (X * m).sum(1) / m.sum(1)

nrm = lambda t: F.normalize(t, dim=-1)

def cos_scores(Q, Dm):          # (q,d) single-vector cosine
    return nrm(Q) @ nrm(Dm).T

def maxsim(Q, Qm, Dm, Dmask):
    """sum over query vectors of max cosine over doc vectors (masked)."""
    Qn, Dn = nrm(Q), nrm(Dm); out = []
    bs = max(1, int(1e8 / (Dn.shape[0] * Qn.shape[1] * Dn.shape[1])))
    for i in range(0, len(Qn), bs):
        s = torch.einsum("qad,nbd->qnab", Qn[i:i+bs], Dn)
        s = s.masked_fill(~Dmask[None, :, None, :], -1e4).max(-1).values
        s = (s * Qm[i:i+bs, None, :]).sum(-1); out.append(s)
    return torch.cat(out)

def metrics(S, name, floats):
    S = S.float(); top = S.topk(10, dim=1).indices.tolist()
    nd10, r10, r1 = [], [], []
    for qi, q in enumerate(qids):
        rel = qrels[q]; ranked = [doc_ids[j] for j in top[qi]]
        dcg = sum(1 / math.log2(k + 2) for k, d in enumerate(ranked) if d in rel)
        idcg = sum(1 / math.log2(k + 2) for k in range(min(len(rel), 10)))
        nd10.append(dcg / idcg); r10.append(len(set(ranked) & rel) / len(rel))
        r1.append(1.0 if ranked[0] in rel else 0.0)
    row = dict(method=name, floats_per_doc=floats, ndcg10=float(np.mean(nd10)),
               recall10=float(np.mean(r10)), hit1=float(np.mean(r1)))
    print(f"{name:52s} {floats:>6} floats  nDCG@10={row['ndcg10']:.3f}  R@10={row['recall10']:.3f}  hit@1={row['hit1']:.3f}", flush=True)
    return row

res = []
kq, kd = torch.ones(len(qids), K), torch.ones(len(doc_ids), K, dtype=torch.bool)
res.append(metrics(cos_scores(sent(q_txt, 256), sent(doc_txt, 256)), "A  MiniLM sentence emb (256 tok)", 384))
res.append(metrics(cos_scores(sent(q_txt, 128), sent(doc_txt, 128)), "A' MiniLM sentence emb (128 tok)", 384))
res.append(metrics(cos_scores(Cq.flatten(1), Cd.flatten(1)), "B  Dragon flatten 3072 (repo way)", 3072))
res.append(metrics(cos_scores(Cq.mean(1), Cd.mean(1)), "C  Dragon mean of 8 slots", 384))
res.append(metrics(maxsim(Cq, kq, Cd, kd), "D  Dragon MaxSim over 8 slots", 3072))
sq, sd = seg_means(Xq, nq), seg_means(Xd, nd)
msq = (sq.abs().sum(-1) > 0).float(); msd = sd.abs().sum(-1) > 0
res.append(metrics(maxsim(sq, msq, sd, msd), "E1 dumb: 8 segment means, MaxSim", 3072))
rq, rd = rand_tok(Xq, nq), rand_tok(Xd, nd)
res.append(metrics(maxsim(rq, (rq.abs().sum(-1) > 0).float(), rd, rd.abs().sum(-1) > 0), "E2 dumb: 8 random tokens, MaxSim", 3072))
res.append(metrics(cos_scores(mean_tok(Xq, nq), mean_tok(Xd, nd)), "E3 dumb: mean of real tokens", 384))
res.append(metrics(maxsim(Xq, mask(nq).float(), Xd, mask(nd)), "F  all tokens MaxSim (no compression)", 49152))

json.dump(dict(results=res, pad_slots=dict(docs=pad_d, queries=pad_q), n_docs=len(doc_ids), n_queries=len(qids)),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w"), indent=1)
print("done", flush=True)
