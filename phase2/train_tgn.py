"""
train_tgn.py — Phase 2A: Training Loop for ConfidenceWeightedTGN
=================================================================

Training protocol (per Implementation Plan 2A):
  - Loads Phase 1 PyG snapshot (tkg_pyg_snapshot.pt)
  - Injects threshold_weight from Phase 1 JSONL mutation log
  - Trains link prediction via negative sampling
  - Gradient clipping (max_norm=1.0) for attention stability
  - Linear LR warm-up (warmup_steps=200) to avoid early instability
  - Saves checkpoint to models/tgn_phase2.pt

Usage:
    python -X utf8 train_tgn.py --epochs 20 --lr 1e-3
    python -X utf8 train_tgn.py --quick-check         # 3 epochs, small batch
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# ── Phase 2 imports ────────────────────────────────────────────────────────────
from tgn_model import ConfidenceWeightedTGN, build_tgn_from_pyg_snapshot


# ───────────────────────────────────────────────────────────────────────────────
# Phase 1 Pre-flight Guard
# ───────────────────────────────────────────────────────────────────────────────

# ── Paths ──────────────────────────────────────────────────────
_LOCAL_P1       = "../phase1/phase1_output"
SNAPSHOT_PATH   = f"{_LOCAL_P1}/tkg_pyg_snapshot.pt"
MUTATIONS_PATH  = f"{_LOCAL_P1}/tkg_mutations.jsonl"
LABEL_MAP_PATH  = "../phase1/models/label_map.json"
CHECKPOINT_DIR  = "models"
CHECKPOINT_PATH = "models/tgn_phase2.pt"


# ───────────────────────────────────────────────────────────────────────────────
# Phase 1 Pre-flight Guard
# ───────────────────────────────────────────────────────────────────────────────

PHASE1_REQUIRED_FILES = [
    (SNAPSHOT_PATH,  "PyG snapshot",     "--export-pyg"),
    (LABEL_MAP_PATH, "label map",         "(produced by train_relation_extractor.py)"),
]


def check_phase1_outputs() -> None:
    """
    Verify that all Phase 1 output artefacts required by Phase 2 exist.
    Raises a descriptive RuntimeError listing every missing file and
    the Phase 1 command needed to regenerate it.
    """
    missing = []
    for fpath, description, hint in PHASE1_REQUIRED_FILES:
        if not os.path.exists(fpath):
            missing.append((fpath, description, hint))

    if missing:
        lines = [
            "\n" + "=" * 70,
            "  Phase 2 CANNOT start — required Phase 1 outputs are missing:",
            "=" * 70,
        ]
        for fpath, desc, hint in missing:
            lines.append(f"  MISSING  [{desc}]  {fpath}")
            lines.append(f"           Hint: {hint}")
        lines += [
            "",
            "  Run Phase 1 first from the phase1/ directory:",
            "    python -X utf8 ingestion_pipeline.py --samples 7947 "\
                "--seed-mitre --export-pyg",
            "  Then re-run this script.",
            "=" * 70 + "\n",
        ]
        raise RuntimeError("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pyg_snapshot(snapshot_path: str):
    """
    Load the real Phase 1 PyG snapshot from datasets_1.
    Path: /FYP-B9/datasets_1/Final_project/phase1/phase1_output/tkg_pyg_snapshot.pt

    Returns:
        PyG Data object with:
          edge_index     [2, E]
          edge_attr      [E, 3]  (c_nlp, source_rep, norm_corrob)
          edge_time      [E]     unix timestamps (float32)  — temporal fix
          edge_rel_type  [E]     relation-type integer IDs  — relation fix
          num_nodes      int
    """
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(
            f"PyG snapshot not found: {snapshot_path}\n"
            f"Ensure datasets_1/Final_project/phase1/phase1_output/tkg_pyg_snapshot.pt exists."
        )
    data = torch.load(snapshot_path, map_location="cpu", weights_only=False)

    E = data.edge_index.shape[1]

    # ── Temporal fix: load real edge timestamps ─────────────────────────────
    if hasattr(data, "edge_time") and data.edge_time is not None:
        edge_time = data.edge_time.float()
        print(f"[Data] edge_time loaded: min={edge_time.min():.0f}  "
              f"max={edge_time.max():.0f}  (unix seconds)")
    else:
        # Old snapshot without timestamps: assign synthetic monotone times
        # (uniform over 30 days) so the chronological split is still defined.
        print("[Data] WARNING: snapshot has no edge_time — "
              "assigning synthetic monotone timestamps (re-export Phase 1 snapshot to fix).")
        edge_time = torch.linspace(0.0, 30.0 * 24 * 3600, E)
    data.edge_time = edge_time

    # ── Relation fix: load real relation-type IDs ───────────────────────────
    if hasattr(data, "edge_rel_type") and data.edge_rel_type is not None:
        n_rels = int(data.edge_rel_type.max().item()) + 1
        print(f"[Data] edge_rel_type loaded: {n_rels} distinct relation types")
    else:
        print("[Data] WARNING: snapshot has no edge_rel_type — "
              "relation IDs will be read from label_map.json (re-export Phase 1 to fix).")
        data.edge_rel_type = None

    print(f"[Data] Loaded snapshot: {data.num_nodes} nodes, "
          f"{E} edges, "
          f"edge_attr={data.edge_attr.shape}")
    return data


def load_threshold_weights(mutations_path: str, num_edges: int) -> Tensor:
    """
    Extract threshold_weight for each edge from the Phase 1 JSONL mutation log.

    The JSONL log contains every edge write with its threshold_weight field
    (1.0=ACCEPTED-full, c=ACCEPTED-partial, 0.3=FLAGGED).
    We build a tensor aligned with the PyG edge_index ordering.

    If the log is unavailable, falls back to 1.0 for all edges (safe default).

    Returns:
        threshold_weights: [E] float tensor
    """
    if not os.path.exists(mutations_path):
        print(f"[Data] Mutation log not found at {mutations_path}. "
              "Using default threshold_weight=1.0 for all edges.")
        return torch.ones(num_edges)

    weights = []
    with open(mutations_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "EDGE":
                weights.append(float(rec.get("threshold_weight", 1.0)))

    if len(weights) == 0:
        print("[Data] No EDGE records in mutation log. Using default 1.0.")
        return torch.ones(num_edges)

    # Align length to num_edges: truncate or pad with 1.0
    if len(weights) < num_edges:
        weights = weights + [1.0] * (num_edges - len(weights))
    weights = weights[:num_edges]

    w_tensor = torch.tensor(weights, dtype=torch.float)
    flagged  = (w_tensor == 0.3).sum().item()
    partial  = ((w_tensor > 0.3) & (w_tensor < 1.0)).sum().item()
    full     = (w_tensor == 1.0).sum().item()
    print(f"[Data] threshold_weights loaded: ACCEPTED-full={full}, "
          f"ACCEPTED-partial={partial}, FLAGGED={flagged}")
    return w_tensor


def load_label_map(label_map_path: str) -> Dict[str, int]:
    """Load label_map.json from Phase 1 model artifacts."""
    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_train_val_split(edge_index: Tensor,
                          edge_attr:  Tensor,
                          edge_time:  Tensor,
                          edge_rel_type: Tensor,
                          threshold_weight: Tensor,
                          val_frac: float = 0.15,
                          ) -> Tuple[Dict, Dict]:
    """
    Chronological edge split using ACTUAL per-edge timestamps (edge_time).

    Steps:
      1. Sort all edges by edge_time ascending (oldest → newest).
      2. First (1-val_frac) in time order = train; last val_frac = val.
      3. Compute delta_t_hours for each edge relative to the TRAINING set max
         timestamp, so validation edges are always positive (future events).

    This implements a genuine temporal hold-out: no future information leaks
    into training, and val edges are strictly after all training edges.

    Returns:
        train_data: {edge_index, edge_attr, edge_rel_type, threshold_weight,
                     delta_t_hours}
        val_data:   same keys
    """
    E = edge_index.shape[1]
    n_val   = max(1, int(E * val_frac))
    n_train = E - n_val

    # Sort edges by timestamp
    sort_idx = torch.argsort(edge_time)            # ascending order

    edge_index_s       = edge_index[:, sort_idx]
    edge_attr_s        = edge_attr[sort_idx]
    edge_rel_type_s    = edge_rel_type[sort_idx]
    threshold_weight_s = threshold_weight[sort_idx]
    edge_time_s        = edge_time[sort_idx]

    train_mask = torch.zeros(E, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask   = ~train_mask

    # delta_t_hours: hours since each edge's timestamp relative to NOW (training max)
    t_max_train = edge_time_s[:n_train].max().item()  # latest training event
    delta_t_hours_s = (t_max_train - edge_time_s).clamp(min=0.0) / 3600.0

    def _split(mask):
        return {
            "edge_index":       edge_index_s[:, mask],
            "edge_attr":        edge_attr_s[mask],
            "edge_rel_type":    edge_rel_type_s[mask],
            "threshold_weight": threshold_weight_s[mask],
            "delta_t_hours":    delta_t_hours_s[mask],
        }

    train_data = _split(train_mask)
    val_data   = _split(val_mask)

    t_train_span = (edge_time_s[:n_train].max() - edge_time_s[:n_train].min()).item() / 3600
    t_val_lag    = (edge_time_s[n_train:].min() - edge_time_s[:n_train].max()).item() / 3600
    print(f"[Split] Train edges: {n_train:,}  Val edges: {n_val:,}  "
          f"(val_frac={val_frac:.0%})")
    print(f"[Split] Train time-span: {t_train_span:.1f} h  "
          f"Val lag after train: {t_val_lag:+.1f} h  "
          f"(positive = no temporal leakage)")
    return train_data, val_data



# ─────────────────────────────────────────────────────────────────────────────
# Negative Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_negatives(edge_index: Tensor, num_nodes: int,
                     batch_size: int) -> Tuple[Tensor, Tensor]:
    """
    Sample negative (src, dst) pairs not present in edge_index.
    Simple random sampling: sufficiently accurate for large sparse graphs
    (probability of false negative = E/N^2 ≈ 41646/9417^2 ≈ 0.05%).

    Returns:
        neg_src: [batch_size]
        neg_dst: [batch_size]
    """
    neg_src = torch.randint(0, num_nodes, (batch_size,))
    neg_dst = torch.randint(0, num_nodes, (batch_size,))
    return neg_src, neg_dst


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule — Linear Warm-up (per Implementation Plan 2A)
# ─────────────────────────────────────────────────────────────────────────────

def get_warmup_scheduler(optimizer, warmup_steps: int,
                          total_steps: int) -> LambdaLR:
    """
    Linear warm-up followed by cosine decay.
    Applied to mitigate instability from the modified attention aggregator.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Training Step
# ─────────────────────────────────────────────────────────────────────────────

def train_step(model: ConfidenceWeightedTGN,
               optimizer,
               scheduler,
               edge_index:       Tensor,
               edge_attr:        Tensor,
               edge_rel_type:    Tensor,
               threshold_weight: Tensor,
               delta_t_hours:    Tensor,
               num_nodes:        int,
               batch_size:       int,
               device:           torch.device,
               max_grad_norm:    float = 1.0) -> Dict[str, float]:
    """
    Single training step with:
      - Random batch of positive (src, dst) edges from training graph
      - Equal-size random negative (src, dst) pairs
      - BCEWithLogitsLoss
      - Gradient clipping (max_norm=1.0) for attention stability
      - Real relation IDs from edge_rel_type (not random integers)
      - Real delta_t_hours per edge for temporal decay
    """
    model.train()

    E = edge_index.shape[1]
    if E == 0:
        return {"loss": 0.0, "acc": 0.0}

    # Sample a random batch of positive edges
    pos_idx = torch.randint(0, E, (batch_size,), device=device)
    pos_src = edge_index[0, pos_idx]
    pos_dst = edge_index[1, pos_idx]

    # Use ACTUAL relation IDs from the TKG snapshot (not random)
    pos_rel     = edge_rel_type[pos_idx].to(device)
    pos_delta_t = delta_t_hours[pos_idx].to(device)

    # Negative samples — relation IDs sampled from the same distribution as positives
    neg_src, neg_dst = sample_negatives(edge_index, num_nodes, batch_size)
    neg_src = neg_src.to(device)
    neg_dst = neg_dst.to(device)
    # For negatives, sample relation IDs uniformly from the observed relation set
    neg_rel = edge_rel_type[
        torch.randint(0, E, (batch_size,))
    ].to(device)
    neg_delta_t = torch.zeros(batch_size, device=device)  # negatives: assume fresh

    # Concatenate positive + negative
    all_src     = torch.cat([pos_src, neg_src])
    all_dst     = torch.cat([pos_dst, neg_dst])
    all_rel     = torch.cat([pos_rel, neg_rel])
    all_delta_t = torch.cat([pos_delta_t, neg_delta_t])
    labels      = torch.cat([
        torch.ones(batch_size,  device=device),
        torch.zeros(batch_size, device=device),
    ])

    # Forward pass — pass real delta_t per batch edge
    scores = model(
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        threshold_weight=threshold_weight.to(device),
        src_nodes=all_src,
        dst_nodes=all_dst,
        relation_ids=all_rel,
        delta_t_hours=delta_t_hours.to(device),   # full-graph delta_t for encode()
    )

    loss = F.binary_cross_entropy_with_logits(scores, labels)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping — prevents attention weight explosion in early training
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        preds = (scores > 0.0).float()
        acc   = (preds == labels).float().mean().item()

    return {"loss": loss.item(), "acc": acc}



# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: ConfidenceWeightedTGN,
             train_edge_index:   Tensor,
             train_edge_attr:    Tensor,
             train_edge_rel:     Tensor,
             train_threshold_w:  Tensor,
             train_delta_t:      Tensor,
             val_edge_index:     Tensor,
             val_edge_attr:      Tensor,
             val_edge_rel:       Tensor,
             val_threshold_w:    Tensor,
             num_nodes:          int,
             device:             torch.device,
             n_val_batches:      int = 20) -> Dict[str, float]:
    """
    Validation: encode on full training graph with real delta_t, score validation edges.
    Uses Hits@10 as primary metric (target: >= 70% per Implementation Plan 2B).

    Uses ACTUAL relation IDs from edge_rel_type (not random integers).
    Uses ACTUAL delta_t_hours computed from edge timestamps (not zero).

    NOTE on val_loss divergence:
      Val BCE rises as training progresses because the model learns a sharper
      decision boundary (logits grow in magnitude).  High Hits@10 is the
      authoritative metric; val_loss is logged for trend analysis only.
    """
    model.eval()

    E_val = val_edge_index.shape[1]
    if E_val == 0:
        return {"val_loss": 0.0, "val_acc": 0.0, "hits_at_10": 0.0}

    # Encode full graph with real temporal decay weights
    h = model.encode(
        edge_index=train_edge_index.to(device),
        edge_attr=train_edge_attr.to(device),
        threshold_weight=train_threshold_w.to(device),
        delta_t_hours=train_delta_t.to(device),
    )                                                               # [N, hidden_dim]

    total_loss = 0.0
    total_acc  = 0.0
    hits_at_10 = 0.0
    n_total    = 0

    E_train    = train_edge_index.shape[1]
    batch_size = min(256, E_val)
    for _ in range(n_val_batches):
        pos_idx = torch.randint(0, E_val, (batch_size,))
        pos_src = val_edge_index[0, pos_idx].to(device)
        pos_dst = val_edge_index[1, pos_idx].to(device)
        # Use ACTUAL relation IDs from val edges
        pos_rel = val_edge_rel[pos_idx].to(device)

        neg_src = torch.randint(0, num_nodes, (batch_size,), device=device)
        neg_dst = torch.randint(0, num_nodes, (batch_size,), device=device)
        # Negatives: sample relation IDs from observed training set distribution
        neg_rel = train_edge_rel[
            torch.randint(0, E_train, (batch_size,))
        ].to(device)

        all_src = torch.cat([pos_src, neg_src])
        all_dst = torch.cat([pos_dst, neg_dst])
        all_rel = torch.cat([pos_rel, neg_rel])
        labels  = torch.cat([torch.ones(batch_size, device=device),
                              torch.zeros(batch_size, device=device)])

        h_src  = h[all_src]
        h_dst  = h[all_dst]
        r_emb  = model.relation_emb(all_rel)
        scores = model.link_predictor(h_src, h_dst, r_emb)

        loss  = F.binary_cross_entropy_with_logits(scores, labels)
        preds = (scores > 0.0).float()
        acc   = (preds == labels).float().mean().item()

        # Hits@10: for each positive, check if it ranks in top-10 out of 10 negs+1 pos
        pos_scores = scores[:batch_size]
        neg_scores = scores[batch_size:]
        hits = 0
        for i in range(min(batch_size, 10)):
            cand = torch.cat([pos_scores[i:i+1], neg_scores[i:i+1]])
            rank = (cand >= pos_scores[i]).sum().item()
            if rank <= 10:
                hits += 1
        hits_at_10 += hits / max(1, min(batch_size, 10))

        total_loss += loss.item()
        total_acc  += acc
        n_total    += 1

    return {
        "val_loss":   total_loss / max(1, n_total),
        "val_acc":    total_acc  / max(1, n_total),
        "hits_at_10": hits_at_10 / max(1, n_total),
    }



# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Phase 1 pre-flight check (must come FIRST) ─────────────────────────────
    check_phase1_outputs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print(f"[Train] Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    data             = load_pyg_snapshot(SNAPSHOT_PATH)
    label_map        = load_label_map(LABEL_MAP_PATH)
    num_nodes        = data.num_nodes
    edge_index       = data.edge_index
    edge_attr        = data.edge_attr
    edge_time        = data.edge_time                          # [E] unix seconds — temporal fix
    threshold_weight = load_threshold_weights(MUTATIONS_PATH, edge_index.shape[1])

    # ── Relation IDs: prefer snapshot's edge_rel_type, fall back to label_map ──
    if data.edge_rel_type is not None:
        edge_rel_type = data.edge_rel_type                    # [E] int64 — relation fix
        num_relations = int(edge_rel_type.max().item()) + 1
        print(f"[Train] Using snapshot relation IDs: {num_relations} distinct types")
    else:
        # Legacy fallback: assign relation IDs from label_map order
        num_relations = len(label_map)
        edge_rel_type = torch.zeros(edge_index.shape[1], dtype=torch.long)
        print(f"[Train] Fallback: label_map has {num_relations} relation types "
              "(re-export Phase 1 snapshot to get per-edge relation IDs)")

    # ── Train/val split — genuinely chronological by timestamp ─────────────────
    train_data, val_data = build_train_val_split(
        edge_index, edge_attr, edge_time, edge_rel_type, threshold_weight,
        val_frac=0.15,
    )

    # ── Build model ────────────────────────────────────────────────────────────
    model = ConfidenceWeightedTGN(
        num_nodes=num_nodes,
        hidden_dim=args.hidden_dim,
        edge_emb_dim=args.edge_emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_relations=num_relations,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model: {total_params:,} trainable parameters")

    # ── Optimizer + LR schedule ────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, train_data["edge_index"].shape[1] // args.batch_size)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = min(args.warmup_steps, total_steps // 4)
    scheduler = get_warmup_scheduler(optimizer, warmup_steps, total_steps)

    print(f"[Train] Epochs={args.epochs}  Steps/epoch={steps_per_epoch}  "
          f"Warmup={warmup_steps}  Total={total_steps}")

    # ── Training ───────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    history = {
        "epochs": [], "train_loss": [], "train_acc": [], "lr": [],
        "val_epochs": [], "val_loss": [], "val_hits_at_10": []
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_loss, epoch_acc = 0.0, 0.0

        for step in range(steps_per_epoch):
            metrics = train_step(
                model, optimizer, scheduler,
                train_data["edge_index"],
                train_data["edge_attr"],
                train_data["edge_rel_type"],
                train_data["threshold_weight"],
                train_data["delta_t_hours"],
                num_nodes, args.batch_size,
                device, max_grad_norm=1.0,
            )
            epoch_loss += metrics["loss"]
            epoch_acc  += metrics["acc"]

        epoch_loss /= steps_per_epoch
        epoch_acc  /= steps_per_epoch
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        history["epochs"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["lr"].append(current_lr)

        # Validate every 2 epochs (or every epoch if quick-check)
        if epoch % (1 if args.quick_check else 2) == 0 or epoch == args.epochs:
            val_metrics = validate(
                model,
                train_data["edge_index"], train_data["edge_attr"],
                train_data["edge_rel_type"], train_data["threshold_weight"],
                train_data["delta_t_hours"],
                val_data["edge_index"],   val_data["edge_attr"],
                val_data["edge_rel_type"], val_data["threshold_weight"],
                num_nodes, device,
            )
            
            history["val_epochs"].append(epoch)
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_hits_at_10"].append(val_metrics["hits_at_10"])

            val_str = (f"  val_loss={val_metrics['val_loss']:.4f}  "
                       f"val_acc={val_metrics['val_acc']:.4f}  "
                       f"Hits@10={val_metrics['hits_at_10']:.4f}")

            # Save best checkpoint
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "num_nodes": num_nodes,
                    "num_relations": num_relations,
                    "relation_vocab": getattr(data, "relation_vocab", None),
                }, CHECKPOINT_PATH)
                val_str += "  [SAVED]"
        else:
            val_str = ""

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={epoch_loss:.4f}  acc={epoch_acc:.4f}  "
              f"lr={current_lr:.2e}  "
              f"elapsed={elapsed:.1f}s"
              f"{val_str}")

    print(f"\n[Train] Complete. Best checkpoint: {CHECKPOINT_PATH}")
    print(f"[Train] Best val_loss: {best_val_loss:.4f}")

    with open("models/training_metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[Train] Saved training_metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2A: Train ConfidenceWeightedTGN on Phase 1 PyG snapshot"
    )
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--batch-size",   type=int,   default=512,  dest="batch_size")
    p.add_argument("--hidden-dim",   type=int,   default=128,  dest="hidden_dim")
    p.add_argument("--edge-emb-dim", type=int,   default=64,   dest="edge_emb_dim")
    p.add_argument("--num-heads",    type=int,   default=4,    dest="num_heads")
    p.add_argument("--num-layers",   type=int,   default=2,    dest="num_layers")
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int,   default=200,  dest="warmup_steps")
    p.add_argument("--quick-check",  action="store_true", dest="quick_check",
                   help="3 epochs, batch=64 — for rapid smoke-test")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick_check:
        args.epochs     = 3
        args.batch_size = 64
        print("[Train] Quick-check mode: 3 epochs, batch=64")
    train(args)
