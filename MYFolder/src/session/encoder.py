from __future__ import annotations

import torch
from torch import nn

EMBED_DIM = 128
N_HEADS = 2
N_LAYERS = 2
SESSION_LENGTH = 20
N_INTENTS = 4
ORTHOGONALITY_WEIGHT = 0.1
COLLAPSE_THRESHOLD = 0.8


class SessionEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        session_length: int = SESSION_LENGTH,
        n_intents: int = N_INTENTS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.session_length = session_length
        self.n_intents = n_intents
        self.position_emb = nn.Embedding(session_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.intent_heads = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_intents)])
        self.weight_head = nn.Linear(embed_dim, n_intents)

    def forward(self, item_embeddings: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, length, _ = item_embeddings.shape
        positions = torch.arange(length, device=item_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        hidden = item_embeddings + self.position_emb(positions)

        # bool, not float, so it matches src_key_padding_mask's dtype -- PyTorch deprecated mixing them
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=item_embeddings.device), diagonal=1
        )

        # A fully-padded row masks every position and produces NaNs in attention softmax;
        # position 0 is always kept attendable as a safe fallback for that edge case.
        key_padding_mask = (mask == 0).clone()
        key_padding_mask[:, 0] = False

        encoded = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)

        lengths = mask.sum(dim=1).clamp(min=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        final_hidden = encoded[torch.arange(batch_size, device=item_embeddings.device), last_idx]

        intents = torch.stack(
            [nn.functional.normalize(head(final_hidden), dim=-1) for head in self.intent_heads], dim=1
        )
        weights = torch.softmax(self.weight_head(final_hidden), dim=-1)
        return intents, weights


def pairwise_head_cosine(intents: torch.Tensor) -> torch.Tensor:
    k = intents.shape[1]
    sims = [(intents[:, i, :] * intents[:, j, :]).sum(dim=-1) for i in range(k) for j in range(i + 1, k)]
    return torch.stack(sims, dim=1)


def orthogonality_penalty(intents: torch.Tensor) -> torch.Tensor:
    return pairwise_head_cosine(intents).mean()
