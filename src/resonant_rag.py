import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import zipfile
import io
from .memory_v3_model import DragonNLP 
from .dragon_quantizer import DragonQuantizer
from sklearn.preprocessing import QuantileTransformer


class ResonantRAG:

    def __init__(self, model_path=None, ratio: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing RAG system (1:{ratio} pooling) on {self.device}...")
        
        # 1. NLP Model and Dragon
        # Use DragonQuantizer to leverage its compression and quantization logic
        self.compressor = DragonQuantizer(model_path=model_path, ratio=ratio)
        self.memory_vectors = self.compressor.memory
        self.memory_texts = self.compressor.texts
        
        # Reference to internal quantizer attributes for easier use
        self.dragon_active = self.compressor.dragon_active
        self.qt = self.compressor.qt  # Quantile Transformer for INT8

        # 2. Synchronization with old names (for backward compatibility)
        self.nlp = self.compressor.teacher
        
        print(f"Dragon Compressor (1:{ratio}) loaded and ready!")

    def _get_vector(self, text):
        """Use compress_text method from DragonQuantizer to get vector."""
        # Vector is already flattened to (K * D_model)
        return self.compressor.compress_text(text)

    def add_memory(self, text):
        """Use add method from DragonQuantizer."""
        self.compressor.add(text)

    def search(self, query, k=3):
        """Search for relevant memories using optimized cosine similarity."""
        if not self.memory_vectors:
            return []
            
        # 1. Get query vector from quantizer
        query_vec = self._get_vector(query).flatten()
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 1e-9:
            query_vec = query_vec / query_norm
            
        # 2. Memory vectors are already in float format for search (dequantized if needed)
        memory_matrix = np.array(self.memory_vectors)
        
        # 3. Execute search (rest of logic is the same)
        memory_norms = np.linalg.norm(memory_matrix, axis=1, keepdims=True)
        memory_norms[memory_norms < 1e-9] = 1.0 
        memory_matrix_norm = memory_matrix / memory_norms
        
        scores = np.dot(memory_matrix_norm, query_vec)
        
        k = min(k, len(self.memory_texts))
        
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.memory_texts[i] for i in top_indices]

    # ==========================================
    # SECURE PERSISTENCE (Updated for quantization)
    # ==========================================

    def save_knowledge_base(self, filename="memory.dragon", use_float16=False, use_int8=False):
        """
        Save knowledge base, enables float16 or INT8 quantization.
        """
        # Use save method from DragonQuantizer
        if use_int8 and not self.qt:
            # If we want INT8, we must first fit QuantileTransformer
            matrix = np.array(self.memory_vectors, dtype=np.float32)
            self.compressor.qt = QuantileTransformer(n_quantiles=256, output_distribution='uniform')
            self.compressor.qt.fit(matrix)  # Fit before saving
            self.qt = self.compressor.qt
            
        return self.compressor.save(filename, use_float16=use_float16, use_int8=use_int8)

    def load_knowledge_base(self, filename="memory.dragon"):
        """
        Load knowledge base, automatically detects INT8 or float format.
        """
        result = self.compressor.load(filename)
        if result[0]:  # If successful, sync memory references
            self.memory_vectors = self.compressor.memory
            self.memory_texts = self.compressor.texts
            self.qt = self.compressor.qt
        return result

    def delete_by_keyword(self, keyword):
        """Delete all vectors and texts that contain a specific keyword."""
        if not keyword:
            return 0, "Keyword cannot be empty."

        keyword = keyword.lower()
        
        # Find indices to keep
        indices_to_keep = [i for i, text in enumerate(self.memory_texts) 
                           if keyword not in text.lower()]
                           
        deleted_count = len(self.memory_texts) - len(indices_to_keep)

        if deleted_count > 0:
            # Create new lists only with indices to keep
            new_texts = [self.memory_texts[i] for i in indices_to_keep]
            new_vectors = [self.memory_vectors[i] for i in indices_to_keep]
            
            # Replace memory
            self.memory_texts[:] = new_texts
            self.memory_vectors[:] = new_vectors
            
            # Sync with compressor
            self.compressor.memory = self.memory_vectors
            self.compressor.texts = self.memory_texts
            
            return deleted_count, f"Successfully deleted {deleted_count} memory units containing '{keyword}'."
        else:
            return 0, f"No memory units found containing '{keyword}'."
