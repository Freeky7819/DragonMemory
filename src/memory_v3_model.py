"""
Memory-v3.4 - Final resonant latent memory head
November 18, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math


class ResonantPointer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 16, depth: int = 6, dropout: float = 0.05):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.final = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.transformer(x)
        return self.final(x).squeeze(-1)


class MultiPhaseResonantPointer(nn.Module):
    """
    v3.7 – Resonant Dragon Phase II
    Multi-phase resonant pointer head.
    """
    def __init__(
        self,
        d_model: int,
        n_phases: int = 4,
        total_depth: int = 12,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_phases = n_phases
        self.total_depth = total_depth

        depth_per_phase = max(1, total_depth // n_phases)

        # Each phase has its own pointer stack
        self.phases = nn.ModuleList([
            ResonantPointer(
                d_model=d_model,
                depth=depth_per_phase,
                dropout=dropout,
            )
            for _ in range(n_phases)
        ])

        # Bottleneck summary for the phase LSTM
        self.phase_projector = nn.Linear(d_model, d_model // 2)
        self.phase_memory = nn.LSTM(
            input_size=d_model // 2,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Confidence gate and per-phase weights
        self.confidence_gate = nn.Linear(d_model, 1)
        self.phase_weights = nn.Parameter(torch.ones(n_phases) / n_phases)

        # Residual feedback strength from phase memory into hidden
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden.shape

        # Accumulated logits over phases
        accumulated_logits = torch.zeros(
            B, T,
            device=hidden.device,
            dtype=hidden.dtype,
        )

        memory_state = None
        current_hidden = hidden

        # Normalized positive phase weights (simplex)
        weights_raw = F.softplus(self.phase_weights)
        weights = weights_raw / (weights_raw.sum() + 1e-6)

        for i, pointer in enumerate(self.phases):
            phase_scores = pointer(current_hidden)
            
            # Per-token confidence from current hidden state (critical for batch=1!)
            gate_raw = self.confidence_gate(current_hidden)          # [B,T,1]
            confidence = torch.sigmoid(gate_raw.squeeze(-1) * 8.0)    # [B,T]
            
            weight = weights[i]
            weighted_scores = phase_scores * confidence * weight
            accumulated_logits = accumulated_logits + weighted_scores
            
            if self.training:
                # Better summary with slight noise
                summary = current_hidden + torch.randn_like(current_hidden) * 0.015
                summary = self.phase_projector(summary.mean(dim=1, keepdim=True))
                lstm_out, memory_state = self.phase_memory(summary, memory_state if i > 0 else None)
                
                feedback = lstm_out.expand(-1, T, -1)
                current_hidden = hidden + self.residual_alpha * feedback

        return accumulated_logits


class NeedleTransformerDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, depth: int = 3, needle_vocab_size: int = 256, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.output_proj = nn.Linear(d_model, needle_vocab_size)

    def forward(self, hidden_needle):
        B, L, D = hidden_needle.shape
        if self.bidirectional:
            mask = None  # all see all - Gemini style
        else:
            mask = torch.triu(torch.ones(L, L, device=hidden_needle.device), diagonal=1).bool()

        x = self.transformer(tgt=hidden_needle, memory=hidden_needle, tgt_mask=mask)
        return self.output_proj(x)


class MemoryV3Model(nn.Module):
    def __init__(
        self,
        base_model_name: str = "microsoft/phi-1_5",
        needle_len: int = 16,
        needle_vocab_size: int = 256,
        pointer_depth: int = 6,
        decoder_bidirectional: bool = False,
        pointer_loss_weight: float = 1.0,
        multi_phase_pointer: bool = False,
        n_pointer_phases: int = 3,
    ):
        super().__init__()
        self.needle_len = needle_len
        self.needle_vocab_size = needle_vocab_size
        self.pointer_loss_weight = pointer_loss_weight

        # Add rope_scaling so model "understands" 32k tokens, even though trained on 2k
        self.lm = AutoModel.from_pretrained(
            base_model_name, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            rope_scaling={"type": "dynamic", "factor": 16.0}  # 2048 * 16 = 32768
        )
        d_model = self.lm.config.hidden_size

        # v3.4 standardni pointer ali v3.7 Dragon multi-phase (switchable)
        if multi_phase_pointer:
            self.pointer = MultiPhaseResonantPointer(
                d_model,
                n_phases=n_pointer_phases,
                total_depth=pointer_depth,
            )
        else:
            self.pointer = ResonantPointer(
                d_model,
                depth=pointer_depth,
            )
        self.decoder = NeedleTransformerDecoder(
            d_model,
            needle_vocab_size=needle_vocab_size,
            bidirectional=decoder_bidirectional
        )

        self.harmonic_weight = nn.Parameter(torch.tensor(1.2))
        self.gamma = 0.0012

    def freeze_lm(self):
        for p in self.lm.parameters():
            p.requires_grad = False

    def harmonic_injection(self, hidden: torch.Tensor):
        B, T, D = hidden.shape
        pos = torch.arange(T, device=hidden.device, dtype=torch.float32)
        signal = torch.exp(-self.gamma * pos) * torch.sin(6.28 * pos + math.pi / 3)
        signal = signal.unsqueeze(0).unsqueeze(-1)
        return hidden + self.harmonic_weight * signal

    def forward(self, input_ids, attention_mask, needle_positions, needle_target_ids):
        B, T = input_ids.shape
        L = self.needle_len

        # 1) Phi-1.5 kot zamrznjen encoder – brez grafa
        with torch.no_grad():
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = outputs.last_hidden_state    # [B, T, D]

        # 2) Resonantna injekcija + glave imajo grad
        hidden = self.harmonic_injection(hidden)

        pointer_logits = self.pointer(hidden)
        safe_pos = needle_positions.clamp(0, T - L)
        loss_pointer = F.cross_entropy(pointer_logits, safe_pos)

        idx = safe_pos.unsqueeze(1) + torch.arange(L, device=input_ids.device)
        idx = idx.clamp(0, T-1)
        hidden_needle = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))

        needle_logits = self.decoder(hidden_needle)
        loss_memory = F.cross_entropy(
            needle_logits.reshape(B * L, -1),
            needle_target_ids.reshape(-1),
        )

        loss = self.pointer_loss_weight * loss_pointer + loss_memory

        with torch.no_grad():
            ptr_acc = (pointer_logits.argmax(-1) == safe_pos).float().mean()
            mem_acc = (needle_logits.argmax(-1) == needle_target_ids).float().mean()
            seq_acc = (needle_logits.argmax(-1) == needle_target_ids).all(-1).float().mean()

        return {
            "loss": loss,
            "loss_pointer": loss_pointer.item(),
            "loss_memory": loss_memory.item(),
            "pointer_accuracy": ptr_acc,
            "memory_token_accuracy": mem_acc,
            "needle_seq_accuracy": seq_acc,
            "pointer_loss_weight": float(self.pointer_loss_weight),
        }

    @torch.no_grad()
    def eval_predicted(self, input_ids, attention_mask, needle_positions, needle_target_ids):
        self.eval()
        B, T = input_ids.shape
        L = self.needle_len

        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        hidden = self.harmonic_injection(hidden)

        pointer_logits = self.pointer(hidden)
        pred_pos = pointer_logits.argmax(-1).clamp(0, T - L)

        idx = pred_pos.unsqueeze(1) + torch.arange(L, device=input_ids.device)
        idx = idx.clamp(0, T-1)
        hidden_needle = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))

        needle_logits = self.decoder(hidden_needle)
        pred_tokens = needle_logits.argmax(-1)

        return {
            "pointer_accuracy_pred": (pred_pos == needle_positions.clamp(0, T - L)).float().mean().item(),
            "memory_token_accuracy_pred": (pred_tokens == needle_target_ids).float().mean().item(),
            "needle_seq_accuracy_pred": (pred_tokens == needle_target_ids).all(-1).float().mean().item(),
        }

    @torch.no_grad()
    def eval_with_reranking(self, input_ids, attention_mask, needle_positions, needle_target_ids, topk: int = 4):
        self.eval()
        B, T = input_ids.shape
        L = self.needle_len

        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        hidden = self.harmonic_injection(hidden)

        pointer_logits = self.pointer(hidden)
        topk_vals, topk_idx = torch.topk(pointer_logits, topk, dim=-1)

        best_seq = torch.zeros(B, device=input_ids.device)
        best_tok = torch.zeros(B, device=input_ids.device)
        best_conf = torch.full((B,), -float('inf'), device=input_ids.device)

        for k in range(topk):
            pos = topk_idx[:, k].clamp(0, T - L)
            idx = pos.unsqueeze(1) + torch.arange(L, device=input_ids.device)
            idx = idx.clamp(0, T-1)
            h_needle = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))

            logits = self.decoder(h_needle)
            logprobs = F.log_softmax(logits, dim=-1)
            confidence = logprobs.gather(-1, needle_target_ids.unsqueeze(-1)).squeeze(-1).mean(-1)

            pred = logits.argmax(-1)
            seq_correct = (pred == needle_target_ids).all(-1).float()
            tok_correct = (pred == needle_target_ids).float().mean(-1)

            better = confidence > best_conf
            best_conf = torch.where(better, confidence, best_conf)
            best_seq = torch.where(better, seq_correct, best_seq)
            best_tok = torch.where(better, tok_correct, best_tok)

        hit_rate = (topk_idx == needle_positions.unsqueeze(1)).any(1).float().mean().item()

        return {
            "pointer_topk_hit_rate": hit_rate,
            "needle_seq_accuracy_rerank": best_seq.mean().item(),
            "needle_token_accuracy_rerank": best_tok.mean().item(),
            "topk": topk,
        }

    @torch.no_grad()
    def preprocess(self, input_ids, attention_mask):
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        hidden = self.harmonic_injection(hidden)
        return hidden.cpu().to(torch.bfloat16)

    @torch.no_grad()
    def forward_cached(self, cached_hidden, needle_positions, needle_target_ids):
        hidden = cached_hidden.to(needle_positions.device)
        B, T, D = hidden.shape
        L = self.needle_len

        pointer_logits = self.pointer(hidden)
        pred_pos = pointer_logits.argmax(-1).clamp(0, T - L)

        idx = pred_pos.unsqueeze(1) + torch.arange(L, device=hidden.device)
        idx = idx.clamp(0, T-1)
        h_needle = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        logits = self.decoder(h_needle)
        return logits.argmax(-1)


# ==========================================
# DRAGON ARCHITECTURE (ADDED AT THE END)
# ==========================================

class DragonNLP(nn.Module):
    """
    Dragon v7: Light Resonant Architecture (Winner 0.904)
    """
    def __init__(self, d_model=384, seq_len=128, ratio=16):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.k = max(1, seq_len // ratio)
        
        # --- v7 CONFIGURATION (2 phases, depth 4) ---
        self.pointer = MultiPhaseResonantPointer(
            d_model=d_model, 
            n_phases=2,       # <--- THIS WAS THE DIFFERENCE
            total_depth=4     # <--- THIS WAS THE DIFFERENCE
        )
        
        # Mixer without Dropout (as it was in v7)
        self.neighbor_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model//32),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=d_model//32),
        )
        
        self.residual = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.pos_bias = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.harmonic_w = nn.Parameter(torch.tensor(0.7))
        self.gamma = 0.0025
        
        # --- ADD THIS LINE (This was the problem!) ---
        self.ln = nn.LayerNorm(d_model)

    def harmonic(self, x):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).float()
        sig = torch.exp(-self.gamma * pos) * torch.sin(6.28 * pos + math.pi/3)
        return x + self.harmonic_w * sig.unsqueeze(0).unsqueeze(-1)

    def compress(self, x):
        h = self.harmonic(x)
        logits = self.pointer(h)
        vals, pos = logits.topk(self.k, dim=1)
        
        m = self.neighbor_mixer(h.transpose(1,2)).transpose(1,2)
        compressed = m.gather(1, pos.unsqueeze(-1).expand(-1,-1,self.d_model))
        
        # Gate
        gate = torch.sigmoid(vals).unsqueeze(-1)
        compressed = compressed * gate
        
        # --- CHANGE THIS LINE ---
        # Previously: compressed = F.layer_norm(...)
        # Now:
        compressed = self.ln(compressed)
        
        positions = pos.float() / self.seq_len
        
        # IMPORTANT: Return 3 things, because v7 needed this for loss!
        return compressed, positions, gate

    def decompress(self, compressed, positions, original_T=None):
        if original_T is None: original_T = self.seq_len
        B, K, D = compressed.shape
        
        summary = compressed.mean(1)
        background = self.residual(summary).unsqueeze(1).expand(-1, original_T, -1)
        background = background + self.pos_bias
        
        recon = background.clone()
        idx = (positions * original_T).long().clamp(0, original_T-1)
        recon.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, D), compressed)
        
        return recon

