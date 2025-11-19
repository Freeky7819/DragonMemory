import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import zipfile
import io
from .memory_v3_model import DragonNLP 

class ResonantRAG:

    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing RAG system on {self.device}...")
        
        # 1. NLP Model
        self.nlp = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Memory Base
        self.memory_vectors = [] 
        self.memory_texts = []
        
        # 3. Dragon - resolve model path
        if model_path is None:
            # Default to models directory relative to project root
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "models", "dragon_pro_1_16.pth")
        
        self.dragon_active = False
        self.init_dragon(model_path)


    def init_dragon(self, path):
        try:
            self.dragon = DragonNLP(d_model=384, seq_len=128, ratio=16).to(self.device)
            state = torch.load(path, map_location=self.device)
            self.dragon.load_state_dict(state)
            self.dragon.eval()
            self.dragon_active = True
            print("Dragon Compressor (1:16) loaded and ready!")
        except FileNotFoundError:
            print(f"WARNING: Could not find '{path}'. Dragon disabled.")


    def _get_vector(self, text):
        emb_list = self.nlp.encode(text, output_value='token_embeddings', convert_to_tensor=True)
        if isinstance(emb_list, list):
            raw = emb_list[0] if len(emb_list) > 0 else torch.zeros(1, 384)
        else:
            raw = emb_list
        
        if not self.dragon_active:
            return raw.mean(dim=0).cpu().numpy()

        padded = torch.zeros(1, 128, 384).to(self.device)
        slen = min(raw.shape[0], 128)
        padded[0, :slen, :] = raw[:slen, :]
        
        with torch.no_grad():
            # V7/V8 return 3 values
            compressed, _, _ = self.dragon.compress(padded)
        
        return compressed.cpu().numpy().flatten()


    def add_memory(self, text):
        vec = self._get_vector(text)
        self.memory_vectors.append(vec)
        self.memory_texts.append(text)


    def search(self, query, k=3):
        if not self.memory_vectors:
            return []
            
        query_vec = self._get_vector(query).flatten()
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 1e-9:
            query_vec = query_vec / query_norm
            
        memory_matrix = np.array(self.memory_vectors)
        memory_norms = np.linalg.norm(memory_matrix, axis=1, keepdims=True)
        memory_norms[memory_norms < 1e-9] = 1.0 
        memory_matrix_norm = memory_matrix / memory_norms
        
        scores = np.dot(memory_matrix_norm, query_vec)
        
        # Limit k if we have fewer memories than k
        k = min(k, len(self.memory_texts))
        
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.memory_texts[i] for i in top_indices]


    # ==========================================
    # SECURE PERSISTENCE
    # ==========================================

    def save_knowledge_base(self, filename="memory.dragon"):
        """
        Save database to ZIP container:
        - vectors.npy (Binary Numpy format - safe and fast)
        - texts.json (Text data)
        """
        if not self.memory_vectors:
            return False, "Database is empty."

        try:
            # 1. Prepare vectors (Numpy Array)
            vectors_array = np.array(self.memory_vectors, dtype=np.float32)
            
            # 2. Prepare texts (List)
            texts_list = self.memory_texts

            # 3. Package into ZIP (no compression for vector speed)
            with zipfile.ZipFile(filename, 'w', zipfile.ZIP_STORED) as zf:
                
                # Save vectors (in memory -> to zip)
                with io.BytesIO() as vec_buffer:
                    np.save(vec_buffer, vectors_array)
                    zf.writestr("vectors.npy", vec_buffer.getvalue())
                
                # Save texts (JSON)
                json_str = json.dumps(texts_list, ensure_ascii=False)
                zf.writestr("texts.json", json_str)

            return True, f"Saved {len(texts_list)} memories to '{filename}'."
        
        except Exception as e:
            return False, str(e)


    def load_knowledge_base(self, filename="memory.dragon"):
        """
        Load database from .dragon file (Safe Load)
        """
        if not os.path.exists(filename):
            return False, "File does not exist."

        try:
            with zipfile.ZipFile(filename, 'r') as zf:
                
                # 1. Load vectors
                with zf.open("vectors.npy") as vec_file:
                    vectors_array = np.load(vec_file)
                    # Convert back to list of arrays (as required by add_memory logic)
                    self.memory_vectors = list(vectors_array)
                
                # 2. Load texts
                with zf.open("texts.json") as txt_file:
                    self.memory_texts = json.load(txt_file)

            return True, f"Successfully loaded {len(self.memory_texts)} memories."
        
        except Exception as e:
            return False, f"Error loading: {e}"

