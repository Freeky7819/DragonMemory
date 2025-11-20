import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import QuantileTransformer
import os
import json
import zipfile
import io
import pickle
from .memory_v3_model import DragonNLP


class DragonQuantizer:
    def __init__(self, model_path=None, ratio=16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = SentenceTransformer("all-MiniLM-L6-v2", device=str(self.device))
        
        # Resolve model path
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "models", "dragon_pro_1_16.pth")
        
        self.dragon = DragonNLP(d_model=384, seq_len=128, ratio=ratio).to(self.device)
        self.dragon.load_state_dict(torch.load(model_path, map_location=self.device))
        self.dragon.eval()
        self.memory = []
        self.texts = []
        self.qt = None  # QuantileTransformer for INT8
        self.dragon_active = True  # Added for compatibility with RAG

    def compress_text(self, text, ratio=None):
        """Compress text to vector using Dragon compression."""
        token_emb = self.teacher.encode(text, output_value="token_embeddings", convert_to_tensor=True)
        if isinstance(token_emb, list):
            token_emb = token_emb[0]
        T = token_emb.shape[0]
        padded = torch.zeros(1, 128, 384).to(self.device)
        slen = min(T, 128)
        padded[0, :slen, :] = token_emb[:slen, :]
        with torch.no_grad():
            out = self.dragon.compress(padded)
            if isinstance(out, (tuple, list)):
                compressed = out[0]
            else:
                compressed = out
        return compressed.cpu().numpy().flatten()

    def add(self, text):
        """Add text to memory."""
        vec = self.compress_text(text)
        self.memory.append(vec)
        self.texts.append(text)

    def quantize(self):
        """Quantize memory vectors to INT8 using QuantileTransformer."""
        # QuantileTransformer must be fitted before quantization!
        if self.qt is None:
            self.qt = QuantileTransformer(n_quantiles=256, output_distribution='uniform')
            matrix = np.array(self.memory, dtype=np.float32)
            self.qt.fit(matrix)

        matrix = np.array(self.memory, dtype=np.float32)
        quantized = (self.qt.transform(matrix) * 255).astype(np.uint8)
        return quantized

    def dequantize(self, quantized_matrix):
        """Dequantize INT8 vectors back to float32."""
        if self.qt is None:
            raise ValueError("Quantizer not initialized. Load a base first.")
        # Return normalized to [0, 1] before inverse transformation
        norm_quantized = quantized_matrix.astype(np.float32) / 255.0
        return self.qt.inverse_transform(norm_quantized)

    def save(self, filename="memory.dragon", use_float16=False, use_int8=False):
        """Save memory base with optional quantization (float16 or INT8)."""
        if not self.memory:
            return False, "Empty memory."
        try:
            with zipfile.ZipFile(filename, 'w', zipfile.ZIP_STORED) as zf:
                if use_int8:
                    quantized = self.quantize()
                    with io.BytesIO() as buf:
                        np.save(buf, quantized)
                        zf.writestr("vectors_int8.npy", buf.getvalue())
                    
                    # Save QuantileTransformer
                    if self.qt is not None:
                        with io.BytesIO() as buf:
                            pickle.dump(self.qt, buf)
                            zf.writestr("quantizer.pkl", buf.getvalue())
                else:
                    dtype = np.float16 if use_float16 else np.float32
                    matrix = np.array(self.memory, dtype=dtype)
                    with io.BytesIO() as buf:
                        np.save(buf, matrix)
                        zf.writestr("vectors.npy", buf.getvalue())
                        
                zf.writestr("texts.json", json.dumps(self.texts, ensure_ascii=False))
            return True, f"Saved {len(self.texts)} items to {filename}"
        except Exception as e:
            return False, str(e)

    def load(self, filename="memory.dragon"):
        """Load memory base, automatically detects INT8 or float format."""
        if not os.path.exists(filename):
            return False, "File not found."
        try:
            with zipfile.ZipFile(filename, 'r') as zf:
                # Load QuantileTransformer
                if "quantizer.pkl" in zf.namelist():
                    with zf.open("quantizer.pkl") as f:
                        self.qt = pickle.load(f)

                if "vectors.npy" in zf.namelist():
                    with zf.open("vectors.npy") as f:
                        self.memory = list(np.load(f))
                elif "vectors_int8.npy" in zf.namelist():
                    with zf.open("vectors_int8.npy") as f:
                        quantized = np.load(f)
                        self.memory = list(self.dequantize(quantized))
                
                with zf.open("texts.json") as f:
                    self.texts = json.load(f)
                    
            return True, f"Loaded {len(self.texts)} items."
        except Exception as e:
            return False, str(e)
