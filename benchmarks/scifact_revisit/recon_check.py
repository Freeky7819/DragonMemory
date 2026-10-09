"""Dragon reconstruction fidelity on real SciFact tokens vs trivial baselines.
Usage: python recon_check.py path/to/scifact
"""
import json, os, sys, torch, torch.nn.functional as F
from sentence_transformers import SentenceTransformer
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = sys.argv[1] if len(sys.argv) > 1 else "scifact"
sys.path.insert(0, REPO)
from src.memory_v3_model import DragonNLP
torch.set_num_threads(2); L, D, K = 128, 384, 8
docs = [json.loads(l) for l in open(f"{DATA}/corpus.jsonl")][:600]
txt = [(d.get("title","")+". "+d["text"]) for d in docs]
qs = [json.loads(l)["text"] for l in open(f"{DATA}/queries.jsonl")][:600]
T = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
dr = DragonNLP(); dr.load_state_dict(torch.load(f"{REPO}/models/dragon_pro_1_16.pth", map_location="cpu")); dr.eval()
def run(texts, label):
    out = T.encode(texts, output_value="token_embeddings", batch_size=64, show_progress_bar=False)
    X = torch.zeros(len(texts), L, D); n = torch.zeros(len(texts), dtype=torch.long)
    for i, t in enumerate(out):
        s = min(t.shape[0], L); X[i, :s] = t[:s]; n[i] = s
    m = torch.arange(L)[None] < n[:, None]
    with torch.no_grad():
        c, p, _ = dr.compress(X); R = dr.decompress(c, p)
    cs = F.cosine_similarity(R, X, dim=-1)
    mean = (X*m[...,None]).sum(1)/m.sum(1,keepdim=True)
    base = F.cosine_similarity(mean[:,None].expand_as(X), X, dim=-1)
    seg = torch.zeros_like(X)
    for i in range(len(X)):
        for ix in torch.tensor_split(torch.arange(int(n[i])), min(K, int(n[i]))): seg[i, ix] = X[i, ix].mean(0)
    sg = F.cosine_similarity(seg, X, dim=-1)
    print(f"{label}: Dragon real tokens={cs[m].mean():.3f} | Dragon incl. padding={cs[~m].nan_to_num(0).mean() if (~m).any() else float('nan'):.3f} | trivial: copy text mean to every token={base[m].mean():.3f} | trivial: 8 segment means={sg[m].mean():.3f}")
run(txt, "documents"); run(qs, "queries")
