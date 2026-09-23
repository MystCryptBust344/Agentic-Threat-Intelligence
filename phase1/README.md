# Phase 1 — Temporal Knowledge Graph (TKG) Ingestion Pipeline

Implements the core data layer for the **Agentic Threat Intelligence** system as defined in the Action Plan (Weeks 1–4).

---

## Architecture

```
  Stream Source (TIRE samples / CTI reports)
          ↓
  TripletExtractor   →  (subject, relation, object, c_NLP)
          ↓
  DualLayerTKG.ingest_triplet()
    ├── Threshold filter  →  ACCEPTED / FLAGGED / REJECTED
    ├── Bounded metrics:
    │     effective_confidence = c_NLP × exp(−λ × Δt)          ∈ [0,1]
    │     threat_reliability   = (ec + source_reputation) / 2  ∈ [0,1]
    └── InMemoryGraphStore (NetworkX/PyG)
              ↓  auto-flush every 30s
    PersistentGraphStore
         ├── Neo4jGraphStore     (live DB — Cypher-native)
         └── JSONLGraphStore     (fallback — append-only log)
```

---

## Files

| File | Purpose |
|---|---|
| `tkg_store.py` | Dual-layer TKG: InMemoryGraphStore + Neo4j/JSONL backends |
| `train_relation_extractor.py` | Fine-tune BERT on TIRE for relation classification |
| `triplet_extractor.py` | Runtime extraction with UQ thresholding |
| `ingestion_pipeline.py` | Streaming simulation driver with Rich telemetry |
| `test_phase1.py` | Full test suite: store, model, e2e |

---

## Quick Start (VS Code)

### Step 0 — Run Tests (db + extractor layers, no model download needed)
```bash
cd phase1
python -X utf8 test_phase1.py --test-db --test-model
```

### Step 1 — Run End-to-End Ingestion Simulation (50 TIRE samples)
```bash
python -X utf8 ingestion_pipeline.py --samples 50
```

### Step 2 — Run with MITRE ATT&CK pre-seeding (recommended for full TKG)
```bash
python -X utf8 ingestion_pipeline.py --samples 100 --seed-mitre
```

### Step 3 — Export PyG graph for Phase 2 TGN training
```bash
python -X utf8 ingestion_pipeline.py --samples 200 --export-pyg
```
Output: `phase1_output/tkg_pyg_snapshot.pt`

### Step 4 — Train the Relation Extractor (GPU recommended)
```bash
# Quick validation (200 samples, 1 epoch)
python -X utf8 train_relation_extractor.py --quick-check

# Full training (7947 samples, 5 epochs, ~hours on CPU)
python -X utf8 train_relation_extractor.py --epochs 5
```

---

## Confidence Thresholding Policy

| c_NLP range | Weight | Status | Action |
|---|---|---|---|
| ≥ 0.90 | 1.0 | ACCEPTED | Immediate ingestion |
| 0.70–0.90 | c | ACCEPTED | Partial weight |
| 0.50–0.70 | 0.3 | FLAGGED | Routed to review queue |
| < 0.50 | — | REJECTED | Logged, excluded from TKG |

---

## Bounded Formula (per Feasibility Study)

```
effective_confidence = c_NLP × exp(−λ × Δt)             λ = 0.01
threat_reliability   = (effective_confidence + reputation) / 2.0
```
All values strictly ∈ [0, 1].

---

## Output Files (phase1_output/)

| File | Description |
|---|---|
| `tkg_mutations.jsonl` | Append-only structural log of all node/edge writes |
| `tkg_node_index.json` | Fast node ID → record lookup table |
| `tkg_pyg_snapshot.pt` | PyG Data object for Phase 2 TGN (if --export-pyg) |
| `models/relation_extractor.pt` | Trained model checkpoint |
| `models/label_map.json` | Relation index ↔ label mapping |
