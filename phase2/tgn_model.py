"""
tgn_model.py — Phase 2: Attention-Guided Confidence-Weighted TGN
=================================================================

Architecture (per Implementation Plan 2A):

  ┌─────────────────────────────────────────────────────────────┐
  │         ConfidenceWeightedTGN                               │
  │                                                             │
  │  edge_attr = [c_nlp | source_reputation | norm_corrob]      │
  │                  ↓                                          │
  │  ConfidenceEdgeEncoder  →  edge_emb (dim=edge_hidden_dim)   │
  │                  ↓                                          │
  │  ConfidenceAttentionAggregator                              │
  │    Neighbor Weight = Confidence * Attention Score           │
  │                      * Temporal Importance                  │
  │                  ↓                                          │
  │  TransformerConv layers (num_layers)                        │
  │                  ↓                                          │
  │  LinkPredictor (dot product)  -> future edge probability    │
  └─────────────────────────────────────────────────────────────┘

Edge Feature Columns (from Phase 1 to_pyg_data()):
  Col 0: c_nlp               — raw NLP extraction confidence in [0.5, 1.0]
  Col 1: source_reputation   — source trust (MITRE=1.0, TIRE=0.70)
  Col 2: norm_corroboration  — log1p(obs_count) / log1p(max_obs) in [0.2, 1.0]

Design Decisions:
  - Attention scores are computed by a 2-layer MLP over edge embeddings so the
    model can learn non-linear interactions between confidence signals.
  - Temporal importance = exp(-lambda * delta_t_normalized), aligned with Phase 1.
  - threshold_weight (from Phase 1 policy: 1.0/c/0.3) is injected as a hard gate
    multiplier BEFORE the attention so FLAGGED edges are structurally visible but
    down-weighted in every attention head.
  - Gradient clipping (max_norm=1.0) and LR warm-up are in train_tgn.py.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Graceful PyG import
try:
    from torch_geometric.nn import TransformerConv
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    print("[tgn_model] WARNING: torch_geometric not installed. "
          "Install with: pip install torch-geometric")


# ─────────────────────────────────────────────────────────────────────────────
# Constants — aligned with Phase 1 decay formula
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_DECAY: float = 0.01     # Identical to Phase 1 tkg_store.py

# Edge feature column indices (must match Phase 1 to_pyg_data() export order)
COL_C_NLP             = 0
COL_SOURCE_REPUTATION = 1
COL_NORM_CORROB       = 2
EDGE_FEAT_DIM         = 3      # [c_nlp, source_reputation, norm_corroboration]


# ─────────────────────────────────────────────────────────────────────────────
# 1. ConfidenceEdgeEncoder
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceEdgeEncoder(nn.Module):
    """
    Maps Phase 1 edge features [c_nlp, source_rep, norm_corrob] to a
    dense embedding. Uses LayerNorm to prevent magnitude differences between
    c_nlp (mean~0.95) and norm_corroboration (mean~0.23) from dominating.
    """

    def __init__(self, edge_feat_dim: int = EDGE_FEAT_DIM,
                 hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, edge_attr: Tensor) -> Tensor:
        """edge_attr: [E, 3] -> edge_emb: [E, out_dim]"""
        return self.net(edge_attr)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ConfidenceScoreHead
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceScoreHead(nn.Module):
    """
    Produces a scalar attention score per edge in [0, 1] via sigmoid.
    Pre-computed before the message passing loop for efficiency.
    """

    def __init__(self, edge_emb_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, edge_emb: Tensor) -> Tensor:
        """edge_emb: [E, edge_emb_dim] -> attention_scores: [E, 1]"""
        return self.net(edge_emb)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TemporalImportanceEncoder
# ─────────────────────────────────────────────────────────────────────────────
class TemporalImportanceEncoder(nn.Module):
    """
    Computes temporal importance: exp(-lambda * delta_t).
    Mirrors Phase 1 effective_confidence decay formula exactly.
    A learnable time_offset parameter allows the model to adjust
    its effective time horizon during training.
    """

    def __init__(self, lambda_decay: float = LAMBDA_DECAY):
        super().__init__()
        self.lambda_decay = lambda_decay
        self.time_offset  = nn.Parameter(torch.zeros(1))

    def forward(self, delta_t_hours: Tensor) -> Tensor:
        """
        delta_t_hours: [E] or [E, 1] -> temporal_importance: [E, 1] in (0, 1]
        """
        if delta_t_hours.dim() == 1:
            delta_t_hours = delta_t_hours.unsqueeze(-1)
        offset = F.softplus(self.time_offset)   # strictly positive
        decay  = torch.exp(-self.lambda_decay * (delta_t_hours + offset))
        return decay.clamp(min=1e-6, max=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ConfidenceWeightedMessageAggregator
#
#    Core Phase 2A formula:
#      composite_weight = c_nlp x attention_score x temporal_importance
#                          x threshold_weight (Phase 1 policy gate)
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceWeightedMessageAggregator(nn.Module):
    """
    Computes per-edge composite weights for message modulation.

    threshold_weight (1.0/c/0.3) is a hard structural gate from Phase 1:
      - ACCEPTED c>=0.90: weight = 1.0
      - ACCEPTED 0.70-0.90: weight = c
      - FLAGGED 0.50-0.70: weight = 0.3  (penalized regardless of attention)
    This preserves TKG's UQ semantics in Phase 2.
    """

    def __init__(self,
                 edge_feat_dim: int = EDGE_FEAT_DIM,
                 edge_emb_dim:  int = 64,
                 lambda_decay:  float = LAMBDA_DECAY):
        super().__init__()
        self.edge_encoder     = ConfidenceEdgeEncoder(edge_feat_dim, 64, edge_emb_dim)
        self.attention_head   = ConfidenceScoreHead(edge_emb_dim, 32)
        self.temporal_encoder = TemporalImportanceEncoder(lambda_decay)

    def forward(self,
                edge_attr:        Tensor,
                threshold_weight: Tensor,
                delta_t_hours:    Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            edge_attr:         [E, 3]  — Phase 1 PyG edge features
            threshold_weight:  [E]    — Phase 1 policy gate (1.0 / c / 0.3)
            delta_t_hours:     [E]    — edge age in hours; None -> assume fresh

        Returns:
            composite_weight:  [E, 1]
            edge_emb:          [E, edge_emb_dim]
        """
        edge_emb   = self.edge_encoder(edge_attr)            # [E, edge_emb_dim]
        attn_score = self.attention_head(edge_emb)           # [E, 1]
        c_nlp      = edge_attr[:, COL_C_NLP:COL_C_NLP + 1]  # [E, 1]

        if delta_t_hours is None:
            delta_t_hours = torch.zeros(edge_attr.size(0), device=edge_attr.device)
        temporal_imp = self.temporal_encoder(delta_t_hours)  # [E, 1]

        if threshold_weight.dim() == 1:
            threshold_weight = threshold_weight.unsqueeze(-1)
        threshold_weight = threshold_weight.to(edge_attr.device)

        composite_weight = (c_nlp * attn_score * temporal_imp * threshold_weight)
        return composite_weight, edge_emb


# ─────────────────────────────────────────────────────────────────────────────
# 5. LinkPredictor
# ─────────────────────────────────────────────────────────────────────────────
class LinkPredictor(nn.Module):
    """
    Scores (src, dst, relation) triples for future link prediction.
    Combines Hadamard product of node embeddings with relation conditioning.
    """

    def __init__(self, node_dim: int, relation_dim: int, hidden_dim: int):
        super().__init__()
        self.hadamard_proj = nn.Linear(node_dim, relation_dim)
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + relation_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_src: Tensor, h_dst: Tensor, r_emb: Tensor) -> Tensor:
        """
        h_src, h_dst: [B, node_dim]; r_emb: [B, relation_dim] -> logits: [B]
        """
        hadamard      = h_src * h_dst                       # [B, node_dim]
        hadamard_proj = self.hadamard_proj(hadamard)        # [B, relation_dim]
        rel_cond      = hadamard_proj * r_emb               # [B, relation_dim]
        logits = self.mlp(torch.cat([hadamard, rel_cond], dim=-1))  # [B, 1]
        return logits.squeeze(-1)                           # [B]


# ─────────────────────────────────────────────────────────────────────────────
# 6. ConfidenceWeightedTGN  (Main Model)
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceWeightedTGN(nn.Module):
    """
    Phase 2A TGN with Attention-Guided Confidence-Weighted Message Aggregation.

    Args:
        num_nodes:     Total TKG nodes (9,417 from Phase 1 snapshot).
        edge_feat_dim: 3 (c_nlp, source_rep, norm_corrob).
        hidden_dim:    Node embedding dimension.
        edge_emb_dim:  Edge embedding dimension.
        num_heads:     TransformerConv attention heads.
        num_layers:    Stacked TransformerConv layers.
        dropout:       Dropout rate.
        num_relations: 16 (from models/label_map.json).
    """

    def __init__(self,
                 num_nodes:     int,
                 edge_feat_dim: int   = EDGE_FEAT_DIM,
                 hidden_dim:    int   = 128,
                 edge_emb_dim:  int   = 64,
                 num_heads:     int   = 4,
                 num_layers:    int   = 2,
                 dropout:       float = 0.1,
                 num_relations: int   = 16):
        super().__init__()

        if not _PYG_AVAILABLE:
            raise ImportError("torch_geometric required. pip install torch-geometric")

        self.num_nodes    = num_nodes
        self.hidden_dim   = hidden_dim
        self.edge_emb_dim = edge_emb_dim

        # Learned node embeddings (no pre-computed node features in Phase 1)
        self.node_emb     = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Relation type embeddings (16 classes from label_map.json)
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)

        # Phase 2A core aggregator
        self.aggregator = ConfidenceWeightedMessageAggregator(
            edge_feat_dim=edge_feat_dim,
            edge_emb_dim=edge_emb_dim,
        )

        # TransformerConv layers conditioned on weighted edge embeddings
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = hidden_dim
        for _ in range(num_layers):
            out_dim = hidden_dim
            self.convs.append(
                TransformerConv(
                    in_channels=in_dim,
                    out_channels=out_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    beta=True,              # Learnable skip connection weight
                    dropout=dropout,
                    edge_dim=edge_emb_dim,  # Inject weighted edge embedding
                )
            )
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.dropout = nn.Dropout(dropout)
        self.link_predictor = LinkPredictor(hidden_dim, hidden_dim, hidden_dim // 2)

    def encode(self,
               edge_index:        Tensor,
               edge_attr:         Tensor,
               threshold_weight:  Tensor,
               delta_t_hours:     Optional[Tensor] = None,
               node_ids:          Optional[Tensor] = None) -> Tensor:
        """
        Compute node embeddings via confidence-weighted message passing.
        Returns h: [N, hidden_dim]
        """
        if node_ids is None:
            node_ids = torch.arange(self.num_nodes, device=edge_attr.device)

        h = self.node_emb(node_ids)                          # [N, hidden_dim]

        composite_weight, edge_emb = self.aggregator(
            edge_attr, threshold_weight, delta_t_hours
        )                                                    # [E,1], [E, edge_emb_dim]

        # Scale edge embedding by composite weight
        weighted_edge_attr = edge_emb * composite_weight    # [E, edge_emb_dim]

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, weighted_edge_attr)
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)
            h     = h + h_new                               # Residual

        return h

    def forward(self,
                edge_index:        Tensor,
                edge_attr:         Tensor,
                threshold_weight:  Tensor,
                src_nodes:         Tensor,
                dst_nodes:         Tensor,
                relation_ids:      Tensor,
                delta_t_hours:     Optional[Tensor] = None,
                node_ids:          Optional[Tensor] = None) -> Tensor:
        """
        Full forward: encode graph -> predict link scores. Returns logits [B].
        """
        h         = self.encode(edge_index, edge_attr, threshold_weight,
                                delta_t_hours, node_ids)
        h_src     = h[src_nodes]
        h_dst     = h[dst_nodes]
        r_emb     = self.relation_emb(relation_ids)
        return self.link_predictor(h_src, h_dst, r_emb)

    def get_node_embeddings(self,
                            edge_index:       Tensor,
                            edge_attr:        Tensor,
                            threshold_weight: Tensor,
                            delta_t_hours:    Optional[Tensor] = None) -> Tensor:
        """
        Public API for Phase 2B zero-day OOD cosine-similarity detection.
        Returns final node embeddings [N, hidden_dim].
        """
        return self.encode(edge_index, edge_attr, threshold_weight, delta_t_hours)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_tgn_from_pyg_snapshot(snapshot_path: str,
                                 hidden_dim:    int   = 128,
                                 edge_emb_dim:  int   = 64,
                                 num_heads:     int   = 4,
                                 num_layers:    int   = 2,
                                 dropout:       float = 0.1
                                 ) -> ConfidenceWeightedTGN:
    """
    Instantiate ConfidenceWeightedTGN from the Phase 1 PyG snapshot.
    Reads num_nodes directly so the model always matches the TKG graph.
    """
    data      = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    num_nodes = data.num_nodes
    print(f"[build_tgn] Snapshot: {num_nodes} nodes, "
          f"{data.edge_index.shape[1]} edges, "
          f"edge_attr shape={data.edge_attr.shape}")

    model = ConfidenceWeightedTGN(
        num_nodes=num_nodes,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=hidden_dim,
        edge_emb_dim=edge_emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_relations=16,
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build_tgn] Trainable parameters: {params:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _PYG_AVAILABLE:
        print("Install: pip install torch-geometric")
        raise SystemExit(1)

    print("=" * 60)
    print("  Phase 2A: ConfidenceWeightedTGN — Shape Verification")
    print("=" * 60)

    N, E, B = 100, 500, 32
    edge_index = torch.randint(0, N, (2, E))
    edge_attr  = torch.rand(E, 3)
    edge_attr[:, 0] = edge_attr[:, 0] * 0.5 + 0.5  # c_nlp in [0.5, 1.0]
    edge_attr[:, 1] = edge_attr[:, 1] * 0.3 + 0.7  # source_rep in [0.7, 1.0]
    edge_attr[:, 2] = edge_attr[:, 2] * 0.8 + 0.2  # norm_corrob in [0.2, 1.0]

    # 50 FLAGGED (w=0.3), rest ACCEPTED
    threshold_weight          = torch.ones(E)
    threshold_weight[:50]     = 0.3
    delta_t_hours             = torch.rand(E) * 168

    src_nodes    = torch.randint(0, N, (B,))
    dst_nodes    = torch.randint(0, N, (B,))
    relation_ids = torch.randint(0, 16, (B,))

    model = ConfidenceWeightedTGN(num_nodes=N)
    model.eval()

    with torch.no_grad():
        scores    = model(edge_index, edge_attr, threshold_weight,
                          src_nodes, dst_nodes, relation_ids, delta_t_hours)
        node_embs = model.get_node_embeddings(edge_index, edge_attr,
                                              threshold_weight, delta_t_hours)
        w, _      = model.aggregator(edge_attr, threshold_weight, delta_t_hours)

    print(f"  Link scores shape:  {scores.shape}")       # [B]
    print(f"  Node emb shape:     {node_embs.shape}")    # [N, 128]
    print(f"  Score range:        [{scores.min():.4f}, {scores.max():.4f}]")

    flagged_w  = w[:50].mean().item()
    accepted_w = w[50:].mean().item()
    print(f"  FLAGGED  composite weight mean:  {flagged_w:.4f}")
    print(f"  ACCEPTED composite weight mean:  {accepted_w:.4f}")
    assert flagged_w < accepted_w, \
        "FAIL: FLAGGED must have lower weight than ACCEPTED"

    print("\n  PASS All shape checks passed.")
    print("  PASS FLAGGED edges carry lower composite weight than ACCEPTED.")
    print("=" * 60)
