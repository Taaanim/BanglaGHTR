"""
Autoregressive Attention Decoder for BANGHTR-X v2.
Implements a causal Transformer decoder with cross-attention to encoder output.
Supports teacher forcing (training) and autoregressive beam search (inference).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from ..positional_encoding import LearnedPositionalEmbedding


class TransformerDecoderBlock(nn.Module):
    """
    Single Pre-Norm Transformer Decoder block with:
    1. Causal self-attention (masked)
    2. Cross-attention to encoder output
    3. Position-wise FFN
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Causal self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # Cross-attention to encoder
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: [B, T_dec, D] decoder input
            memory: [B, T_enc, D] encoder output
            tgt_mask: [T_dec, T_dec] causal attention mask
            memory_key_padding_mask: [B, T_enc] True for padded encoder positions
        Returns:
            [B, T_dec, D]
        """
        # Causal self-attention
        norm_tgt = self.norm1(tgt)
        sa_out, _ = self.self_attn(norm_tgt, norm_tgt, norm_tgt, attn_mask=tgt_mask)
        tgt = tgt + sa_out

        # Cross-attention to encoder
        norm_tgt = self.norm2(tgt)
        ca_out, _ = self.cross_attn(
            norm_tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + ca_out

        # FFN
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt


class AttentionDecoder(nn.Module):
    """
    Autoregressive Transformer decoder for Bengali HTR.
    During training: uses teacher forcing with cross-entropy + label smoothing.
    During inference: supports greedy decode and beam search.
    """
    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        label_smoothing: float = 0.1,
        pad_idx: int = 1,  # <PAD> index in tokenizer
        bos_idx: int = 3,  # <BOS> index we'll add
        eos_idx: int = 4   # <EOS> index we'll add
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Token embedding + positional encoding
        self.token_embedding = nn.Embedding(num_classes, d_model, padding_idx=pad_idx)
        self.pos_embedding = LearnedPositionalEmbedding(max_seq_len, d_model, dropout)
        self.embed_scale = d_model ** 0.5

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_classes)

        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing
        )

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates an upper-triangular causal mask for self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask  # True = masked positions

    def forward(
        self,
        encoder_out: torch.Tensor,
        target_tokens: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forced forward pass.

        Args:
            encoder_out: [B, T_enc, D] from encoder
            target_tokens: [B, T_dec] target token indices (with BOS prepended)
            memory_key_padding_mask: [B, T_enc]
        Returns:
            logits: [B, T_dec, num_classes]
            loss: scalar cross-entropy loss
        """
        # Shift targets: input = tokens[:-1], gold = tokens[1:]
        decoder_input = target_tokens[:, :-1]  # [B, T-1]
        decoder_gold = target_tokens[:, 1:]     # [B, T-1]

        # Embed tokens
        tgt = self.token_embedding(decoder_input) * self.embed_scale  # [B, T-1, D]
        tgt = self.pos_embedding(tgt)

        # Causal mask
        tgt_len = decoder_input.size(1)
        causal_mask = self._generate_causal_mask(tgt_len, decoder_input.device)

        # Run decoder layers
        for layer in self.layers:
            tgt = layer(tgt, encoder_out, tgt_mask=causal_mask,
                       memory_key_padding_mask=memory_key_padding_mask)

        tgt = self.final_norm(tgt)
        logits = self.output_proj(tgt)  # [B, T-1, num_classes]

        # Compute loss
        loss = self.criterion(
            logits.reshape(-1, self.num_classes),
            decoder_gold.reshape(-1)
        )

        return logits, loss

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        max_len: int = 150,
        memory_key_padding_mask: torch.Tensor = None,
        return_scores: bool = False,
    ):
        """
        Autoregressive greedy decoding.

        Args:
            encoder_out: [B, T_enc, D]
            max_len: maximum output sequence length
            return_scores: if True, returns (results, list_of_confidence_scores)
        Returns:
            List of decoded token index lists for each batch sample, or (results, scores) if return_scores=True
        """
        B = encoder_out.size(0)
        device = encoder_out.device

        # Start with BOS token
        ys = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        token_probs_list = [[] for _ in range(B)]

        for _ in range(max_len):
            tgt = self.token_embedding(ys) * self.embed_scale
            tgt = self.pos_embedding(tgt)
            causal_mask = self._generate_causal_mask(ys.size(1), device)

            for layer in self.layers:
                tgt = layer(tgt, encoder_out, tgt_mask=causal_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

            tgt = self.final_norm(tgt)
            logits = self.output_proj(tgt[:, -1, :])  # [B, num_classes]
            probs = logits.softmax(dim=-1)
            next_token = logits.argmax(dim=-1)  # [B]

            if return_scores:
                for b_i in range(B):
                    if not finished[b_i]:
                        token_probs_list[b_i].append(float(probs[b_i, next_token[b_i]].item()))

            # Check for EOS
            finished = finished | (next_token == self.eos_idx)
            next_token = next_token.masked_fill(finished, self.pad_idx)

            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # Convert to list of lists, removing BOS/EOS/PAD
        results = []
        for seq in ys:
            tokens = []
            for tok in seq.tolist():
                if tok == self.bos_idx:
                    continue
                if tok == self.eos_idx:
                    break
                if tok == self.pad_idx:
                    continue
                tokens.append(tok)
            results.append(tokens)

        if return_scores:
            scores = []
            for p_list in token_probs_list:
                scores.append(float(np.mean(p_list)) if p_list else 0.0)
            return results, scores

        return results

    @torch.no_grad()
    def greedy_decode_with_margin(
        self,
        encoder_out: torch.Tensor,
        max_len: int = 150,
        memory_key_padding_mask: torch.Tensor = None,
        repetition_penalty: float = 0.40,
    ):
        """
        Greedy autoregressive decoding that returns (tokens, score) where
        `score` is a calibrated 0–1 confidence based on logit MARGIN
        (logit(argmax) - logit(second)) and a repetition penalty.

        Why margin instead of softmax-of-argmax?
        A confident model commits hard to one token (large gap to #2). A
        confused one has a near-tied runner-up. Softmax-of-argmax can be
        ~0.7 just from softmax temperature even when the runner-up is
        almost as likely; margin directly measures commitment.

        Repetition penalty:
        For each emitted step t, if next_token == previous emitted token,
        the score for that step is multiplied by (1 - repetition_penalty).
        This kills the runaway-repeat pathology (e.g. "ককককক") that the
        original softmax-mean confidence could not detect.
        """
        import numpy as np  # local to avoid touching module top-level imports

        B = encoder_out.size(0)
        device = encoder_out.device

        ys = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        per_step_margin = [[] for _ in range(B)]
        emitted_tokens_per_batch = [[] for _ in range(B)]

        for _ in range(max_len):
            tgt = self.token_embedding(ys) * self.embed_scale
            tgt = self.pos_embedding(tgt)
            causal_mask = self._generate_causal_mask(ys.size(1), device)

            for layer in self.layers:
                tgt = layer(tgt, encoder_out, tgt_mask=causal_mask,
                            memory_key_padding_mask=memory_key_padding_mask)

            tgt = self.final_norm(tgt)
            logits = self.output_proj(tgt[:, -1, :])  # [B, num_classes]

            # Top-2 logits → margin (logit scale, in [0, +inf))
            top2_vals, top2_idx = logits.topk(2, dim=-1)
            margin = (top2_vals[:, 0] - top2_vals[:, 1]).clamp(min=0.0)
            # squash margin into [0,1] via 1 - exp(-margin); margin of ~3.0 → 0.95
            margin_conf = 1.0 - torch.exp(-margin)
            next_token = top2_idx[:, 0]

            for b_i in range(B):
                if finished[b_i]:
                    continue
                step_score = float(margin_conf[b_i].item())
                # Repetition penalty
                if (
                    emitted_tokens_per_batch[b_i]
                    and emitted_tokens_per_batch[b_i][-1] == int(next_token[b_i].item())
                ):
                    step_score *= (1.0 - repetition_penalty)
                per_step_margin[b_i].append(step_score)
                emitted_tokens_per_batch[b_i].append(int(next_token[b_i].item()))

            finished = finished | (next_token == self.eos_idx)
            next_token = next_token.masked_fill(finished, self.pad_idx)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # Convert to token lists (strip BOS/EOS/PAD)
        results = []
        for seq in ys:
            tokens = []
            for tok in seq.tolist():
                if tok == self.bos_idx:
                    continue
                if tok == self.eos_idx:
                    break
                if tok == self.pad_idx:
                    continue
                tokens.append(tok)
            results.append(tokens)

        scores = []
        for margin_list in per_step_margin:
            if not margin_list:
                scores.append(0.0)
            else:
                # Geometric mean is much harsher on any single weak step
                # than arithmetic mean — appropriate for confidence.
                m = np.array(margin_list, dtype=np.float64)
                # avoid log(0)
                m = np.clip(m, 1e-6, 1.0)
                scores.append(float(np.exp(np.mean(np.log(m)))))
        return results, scores


    def sample_decode(
        self,
        encoder_out: torch.Tensor,
        temperature: float = 1.0,
        max_len: int = 150,
        memory_key_padding_mask: torch.Tensor = None
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """
        Sampling-based decoding for SCST/RL training.
        Returns both sampled sequences and their log-probabilities.

        Args:
            encoder_out: [B, T_enc, D]
            temperature: sampling temperature (higher = more diverse)
            max_len: maximum output length
        Returns:
            sampled_seqs: List of decoded token lists
            log_probs: [B] total log probability of each sampled sequence
        """
        B = encoder_out.size(0)
        device = encoder_out.device

        ys = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        total_log_probs = torch.zeros(B, device=device)

        for _ in range(max_len):
            tgt = self.token_embedding(ys) * self.embed_scale
            tgt = self.pos_embedding(tgt)
            causal_mask = self._generate_causal_mask(ys.size(1), device)

            for layer in self.layers:
                tgt = layer(tgt, encoder_out, tgt_mask=causal_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

            tgt = self.final_norm(tgt)
            logits = self.output_proj(tgt[:, -1, :])  # [B, num_classes]

            # Temperature-scaled sampling
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

            # Accumulate log probs for non-finished sequences
            log_p = F.log_softmax(logits, dim=-1)
            token_log_probs = log_p.gather(1, next_token.unsqueeze(1)).squeeze(-1)
            total_log_probs += token_log_probs * (~finished).float()

            finished = finished | (next_token == self.eos_idx)
            next_token = next_token.masked_fill(finished, self.pad_idx)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # Convert to list of lists
        results = []
        for seq in ys:
            tokens = []
            for tok in seq.tolist():
                if tok == self.bos_idx:
                    continue
                if tok == self.eos_idx:
                    break
                if tok == self.pad_idx:
                    continue
                tokens.append(tok)
            results.append(tokens)

        return results, total_log_probs

    @torch.no_grad()
    def beam_search(
        self,
        encoder_out: torch.Tensor,
        beam_width: int = 5,
        max_len: int = 150,
        memory_key_padding_mask: torch.Tensor = None,
        length_penalty: float = 0.6
    ) -> List[List[int]]:
        """
        Beam search decoding for highest quality inference.

        Args:
            encoder_out: [B, T_enc, D] — processes one sample at a time internally
            beam_width: number of beams
            max_len: maximum output length
            length_penalty: length normalization exponent
        Returns:
            List of decoded token lists for each batch sample
        """
        B = encoder_out.size(0)
        device = encoder_out.device
        all_results = []

        for b in range(B):
            # Single sample encoder output [1, T_enc, D] -> expand to beam_width
            enc_b = encoder_out[b:b+1].expand(beam_width, -1, -1)  # [beam, T_enc, D]
            mask_b = None
            if memory_key_padding_mask is not None:
                mask_b = memory_key_padding_mask[b:b+1].expand(beam_width, -1)

            # Initialize beams: (tokens, score)
            ys = torch.full((beam_width, 1), self.bos_idx, dtype=torch.long, device=device)
            scores = torch.zeros(beam_width, device=device)
            scores[1:] = -1e9  # Only first beam is active initially

            finished_beams = []

            for step in range(max_len):
                tgt = self.token_embedding(ys) * self.embed_scale
                tgt = self.pos_embedding(tgt)
                causal_mask = self._generate_causal_mask(ys.size(1), device)

                for layer in self.layers:
                    tgt = layer(tgt, enc_b, tgt_mask=causal_mask,
                               memory_key_padding_mask=mask_b)

                tgt = self.final_norm(tgt)
                logits = self.output_proj(tgt[:, -1, :])  # [beam, vocab]
                log_probs = F.log_softmax(logits, dim=-1)  # [beam, vocab]

                # Expand scores: [beam, vocab]
                next_scores = scores.unsqueeze(1) + log_probs  # [beam, vocab]

                # Flatten and get top-k
                vocab_size = log_probs.size(-1)
                flat_scores = next_scores.view(-1)  # [beam * vocab]
                top_scores, top_indices = flat_scores.topk(beam_width, dim=0)

                beam_indices = top_indices // vocab_size
                token_indices = top_indices % vocab_size

                # Reorder beams
                ys = torch.cat([ys[beam_indices], token_indices.unsqueeze(1)], dim=1)
                scores = top_scores

                # Check for finished beams
                active_mask = token_indices != self.eos_idx
                for i in range(beam_width):
                    if not active_mask[i]:
                        length = ys.size(1) - 1  # exclude BOS
                        norm_score = scores[i].item() / (length ** length_penalty)
                        finished_beams.append((ys[i].clone(), norm_score))
                        scores[i] = -1e9  # deactivate

                if len(finished_beams) >= beam_width:
                    break

            # If no beam finished, use the best active one
            if not finished_beams:
                for i in range(beam_width):
                    length = ys.size(1) - 1
                    norm_score = scores[i].item() / max(1, length ** length_penalty)
                    finished_beams.append((ys[i], norm_score))

            # Select best beam
            best_beam = max(finished_beams, key=lambda x: x[1])
            tokens = []
            for tok in best_beam[0].tolist():
                if tok == self.bos_idx:
                    continue
                if tok == self.eos_idx:
                    break
                if tok == self.pad_idx:
                    continue
                tokens.append(tok)
            all_results.append(tokens)

        return all_results
