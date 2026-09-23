"""
ingestion_pipeline.py — Phase 1: Streaming Ingestion Simulation
================================================================
Simulates a continuous stream of CTI reports feeding the TKG.

Architecture:
  ┌───────────────────────────────────────────────────────────────┐
  │  Stream Source (TIRE samples / synthetic threat reports)      │
  │         ↓  text passages                                      │
  │  TripletExtractor  →  (s, r, o, c_NLP)                       │
  │         ↓                                                     │
  │  DualLayerTKG.ingest_triplet()                                │
  │    ├── threshold filter → ACCEPTED / FLAGGED / REJECTED       │
  │    ├── compute bounded metrics:                               │
  │    │     effective_confidence = c_NLP * exp(-λ * Δt)         │
  │    │     threat_reliability   = (ec + reputation) / 2        │
  │    └── write → InMemoryGraphStore → auto-flush → JSONL/Neo4j  │
  │         ↓                                                     │
  │  Telemetry Console + final report + PyG graph export          │
  └───────────────────────────────────────────────────────────────┘

Run modes:
    python ingestion_pipeline.py                    # Simulate 50 samples
    python ingestion_pipeline.py --samples 200      # More samples
    python ingestion_pipeline.py --seed-mitre       # Pre-seed MITRE first
    python ingestion_pipeline.py --export-pyg       # Export PyG graph at end
"""

import argparse
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from tkg_store import (
    DualLayerTKG,
    TKGNode,
    build_persistent_store,
    SOURCE_REPUTATION,
)
from triplet_extractor import TripletExtractor, ExtractedTriplet

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
TIRE_JSON     = "../TIRE/dnrti_aug_stix2_je.json"
MITRE_JSON    = "../enterprise-attack.json"
OUTPUT_DIR    = "phase1_output"
FLUSH_INTERVAL = 10        # Flush every N seconds (reduced for demo)
LAMBDA_DECAY  = 0.01       # Must match tkg_store.py
console       = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _make_summary_table(accepted: int, flagged: int, rejected: int,
                         total_nodes: int, total_edges: int,
                         elapsed: float,
                         conf_stats: dict = None,
                         dup_merged: int = 0) -> Table:
    t = Table(title="Phase 1 v2 — Ingestion Summary", show_header=True,
              header_style="bold magenta", border_style="bright_blue")
    t.add_column("Metric",  style="cyan",  no_wrap=True)
    t.add_column("Value",   style="green", justify="right")
    total = accepted + flagged + rejected
    t.add_row("Triplets ACCEPTED",       str(accepted))
    t.add_row("Triplets FLAGGED",        f"[yellow]{flagged}[/yellow]")
    t.add_row("Triplets REJECTED",       f"[red]{rejected}[/red]")
    t.add_row("Accept rate",             f"{100*accepted/max(total,1):.1f}%")
    t.add_row("Flag rate",               f"{100*flagged/max(total,1):.1f}%")
    t.add_row("Reject rate",             f"{100*rejected/max(total,1):.1f}%")
    t.add_row("TKG Nodes",               str(total_nodes))
    t.add_row("TKG Edges (unique)",      str(total_edges))
    t.add_row("Duplicate edges merged",  f"[dim]{dup_merged}[/dim]")
    t.add_row("Elapsed (s)",             f"{elapsed:.2f}")
    t.add_row("Throughput (triplets/s)", f"{total/max(elapsed,1):.1f}")
    if conf_stats:
        t.add_row("", "")
        t.add_row("c_NLP min",           f"{conf_stats['min']:.4f}")
        t.add_row("c_NLP max",           f"{conf_stats['max']:.4f}")
        t.add_row("c_NLP mean",          f"{conf_stats['mean']:.4f}")
        t.add_row("c_NLP std",           f"{conf_stats['std']:.4f}")
        t.add_row("Unique c_NLP (3dp)",  str(conf_stats['unique']))
        collapsed = conf_stats.get('std', 0) < 0.01
        t.add_row("UQ signal",
                  "[red]COLLAPSED[/red]" if collapsed else "[green]VARYING[/green]")
    return t


def _triplet_row(t: ExtractedTriplet) -> str:
    """One-line console log for a triplet."""
    color = {"ACCEPTED": "green", "FLAGGED": "yellow", "REJECTED": "red"}[t.status]
    return (
        f"[{color}][{t.status}][/{color}] "
        f"[bold]{t.subject}[/bold] ({t.subject_label}) "
        f"—[{t.relation}]→ "
        f"[bold]{t.object_}[/bold] ({t.object_label}) "
        f"c_NLP={t.c_nlp:.3f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Node Auto-registration
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_nodes(tkg: DualLayerTKG, triplet: ExtractedTriplet,
                  known_nodes: set) -> None:
    """Register subject and object nodes if not already in the graph."""
    for entity_name, entity_label in [
        (triplet.subject, triplet.subject_label),
        (triplet.object_, triplet.object_label)
    ]:
        node_id = f"{entity_label}::{entity_name}"
        if node_id not in known_nodes:
            tkg.ingest_node(TKGNode(
                node_id=node_id,
                label=entity_label,
                name=entity_name,
                properties={"auto_registered": True}
            ))
            known_nodes.add(node_id)


# ──────────────────────────────────────────────────────────────────────────────
# MITRE Seeding via DualLayerTKG (both layers)
# ──────────────────────────────────────────────────────────────────────────────
def _seed_mitre_through_tkg(tkg: DualLayerTKG, mitre_path: str) -> None:
    """
    Parse enterprise-attack.json and seed ALL MITRE nodes + edges through the
    DualLayerTKG so that BOTH InMemoryGraphStore and the persistent backend
    receive every node and edge.

    Previously, seeding went directly to the persistent store only, leaving
    the in-memory graph empty of MITRE topology. This caused all subgraph
    queries and PyG exports to miss the MITRE knowledge.
    """
    from tkg_store import (
        TKGNode, MITRE_TYPE_MAP, MITRE_REL_MAP, SOURCE_REPUTATION
    )
    import time as _time

    with open(mitre_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects", [])

    # Build id_map for relationship resolution
    id_map: dict = {}

    # Pass 1 — Nodes
    nodes_written = 0
    for obj in objects:
        obj_type = obj.get("type", "")
        label    = MITRE_TYPE_MAP.get(obj_type)
        if label is None:
            continue
        # Use 'LABEL::uuid' format so the '::'-based type extractor
        # (node_id.split('::')[0]) returns the entity type label, not
        # the raw UUID. This is the same convention TIRE nodes use
        # (e.g., 'APT::admin@338'). Without this, all MITRE nodes
        # land in 'Unknown->Unknown' because raw UUIDs have no '::'.
        prefixed_id = f"{label}::{obj['id']}"
        node = TKGNode(
            node_id=prefixed_id,
            label=label,
            name=obj.get("name", obj.get("id", "unknown")),
            properties={"description": obj.get("description", "")[:200],
                        "source": "mitre-attack"},
            timestamp=_time.time()
        )
        tkg.ingest_node(node)          # → InMemoryGraphStore + pending flush
        id_map[obj["id"]] = prefixed_id  # map raw UUID → prefixed node_id
        nodes_written += 1

    # Pass 2 — Edges (relationships through ingest_triplet for full UQ pipeline)
    edges_written = 0
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        rel_type = MITRE_REL_MAP.get(obj.get("relationship_type", ""))
        if not rel_type:
            continue
        src_id = obj.get("source_ref", "")
        tgt_id = obj.get("target_ref", "")
        if src_id not in id_map or tgt_id not in id_map:
            continue
        # id_map[raw_uuid] = 'LABEL::uuid' — use prefixed IDs for both nodes
        tkg.ingest_triplet(
            source_id=id_map[src_id],
            relation=rel_type,
            target_id=id_map[tgt_id],
            c_nlp=1.0,
            source_name="mitre-attack",
            timestamp=_time.time()
        )
        edges_written += 1

    console.print(f"  MITRE parsed: [cyan]{nodes_written:,}[/cyan] nodes + "
                  f"[cyan]{edges_written:,}[/cyan] edges queued")

def _show_decay_demo():
    console.print("\n[bold cyan]Confidence Decay Demo  (lambda=0.01/hour, starting c_NLP=0.92)[/bold cyan]")
    t = Table(show_header=True, header_style="bold", border_style="dim")
    t.add_column("Delta-t (hours)", justify="right")
    t.add_column("effective_confidence", justify="right")
    t.add_column("threat_reliability (rep=0.80)", justify="right")
    lambda_h = 0.01   # per-hour for display; tkg_store uses 0.01/3600 per-second
    for hours in [0, 1, 6, 12, 24, 72, 168, 336, 720]:
        ec = 0.92 * math.exp(-lambda_h * hours)
        ec = max(0.0, min(1.0, ec))
        tr = (ec + 0.80) / 2.0
        t.add_row(str(hours), f"{ec:.4f}", f"{tr:.4f}")
    console.print(t)



# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(args):
    console.print(Panel.fit(
        "[bold magenta]Phase 1 — Agentic TI Ingestion Pipeline[/bold magenta]\n"
        "Temporal Knowledge Graph Construction with Uncertainty Quantification",
        border_style="magenta"
    ))

    # ── Step 0: Setup Store + TKG ─────────────────────────────────────────────
    persistent = build_persistent_store(
        output_dir=args.output_dir,
        neo4j_uri=args.neo4j_uri or None
    )
    tkg = DualLayerTKG(persistent, flush_interval=FLUSH_INTERVAL)
    extractor = TripletExtractor()

    console.print(f"\n[bold]Store Backend:[/bold] [green]{type(persistent).__name__}[/green]")
    console.print(f"[bold]Extractor Mode:[/bold] [green]{'Model' if extractor._model_loaded else 'Heuristic fallback'}[/green]")

    # ── Step 1: Seed MITRE ATT&CK through DualLayerTKG ───────────────────────
    if args.seed_mitre and os.path.exists(MITRE_JSON):
        console.print("\n[bold yellow]Seeding MITRE ATT&CK through TKG (nodes + edges -> both layers)...[/bold yellow]")
        t0 = time.time()
        _seed_mitre_through_tkg(tkg, MITRE_JSON)
        # Flush MITRE batch immediately so it persists before streaming starts
        tkg.flush()
        console.print(f"  [green]done[/green] MITRE seeded in {time.time()-t0:.1f}s  "
                      f"| in-memory: {tkg.memory.stats['nodes']} nodes, "
                      f"{tkg.memory.stats['edges']} edges")

    # ── Step 2: Load TIRE stream ──────────────────────────────────────────────
    console.print(f"\n[bold yellow]⟳ Loading TIRE dataset stream...[/bold yellow]")
    with open(TIRE_JSON, "r", encoding="utf-8") as f:
        tire_samples = json.load(f)

    samples = tire_samples[:args.samples]
    console.print(f"  Loaded [bold]{len(samples):,}[/bold] CTI report samples for simulation.")

    # ── Step 3: Show decay math ───────────────────────────────────────────────
    _show_decay_demo()

    # ── Step 4: Streaming ingestion ───────────────────────────────────────────
    console.print(f"\n[bold yellow]⟳ Starting streaming ingestion...[/bold yellow]")

    accepted_count = 0
    flagged_count  = 0
    rejected_count = 0
    known_nodes: set = set()
    all_confs: list = []       # RAW c_NLP per triplet (before any weighting)
    rejected_samples: list = []  # for REJECTED spot-check (up to 10)

    t_start = time.time()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Ingesting CTI reports…", total=len(samples))

        for idx, sample in enumerate(samples):
            # Use ground-truth entity annotations from TIRE for this demo
            triplets = extractor.extract_from_tire_sample(sample)

            # Simulate time progression: each sample = +5 minutes
            sim_time = t_start + idx * 300

            for tri in triplets:
                # Ensure subject + object nodes exist
                _ensure_nodes(tkg, tri, known_nodes)

                src_id = f"{tri.subject_label}::{tri.subject}"
                tgt_id = f"{tri.object_label}::{tri.object_}"

                # Pass RAW c_nlp to ingest_triplet (threshold applied once inside)
                edge = tkg.ingest_triplet(
                    source_id=src_id,
                    relation=tri.relation,
                    target_id=tgt_id,
                    c_nlp=tri.c_nlp,    # tri.c_nlp is now RAW (not weighted)
                    source_name="tire",
                    timestamp=sim_time
                )

                # Count from actual TKG routing (not from tri.status)
                if edge is None:
                    rejected_count += 1
                    if len(rejected_samples) < 10:
                        rejected_samples.append(tri)
                elif edge.threshold_weight == 0.3:
                    flagged_count += 1
                else:
                    accepted_count += 1

                # Collect RAW confidence for histogram (tri.c_nlp is now raw)
                if tri.c_nlp > 0:
                    all_confs.append(tri.c_nlp)

                # Verbose output for first 5 samples
                if idx < 5:
                    console.print(f"  {_triplet_row(tri)}")

            progress.advance(task)

    elapsed = time.time() - t_start

    # ── Step 5: Final flush ───────────────────────────────────────────────────
    console.print("\n[bold yellow]⟳ Flushing final batch to persistent store…[/bold yellow]")
    tkg.flush()

    # ── Step 6: Confidence stats ──────────────────────────────────────────────
    import numpy as np
    conf_stats = None
    if all_confs:
        ca = np.array(all_confs)
        conf_stats = {
            "min": float(ca.min()),
            "max": float(ca.max()),
            "mean": float(ca.mean()),
            "std": float(ca.std()),
            "unique": len(set(ca.round(3))),
        }

    # ── Step 7: Summary Table ─────────────────────────────────────────────────
    mem_stats = tkg.memory.stats
    console.print("\n")
    console.print(_make_summary_table(
        accepted_count, flagged_count, rejected_count,
        mem_stats["nodes"], mem_stats["edges"], elapsed, conf_stats,
        dup_merged=mem_stats.get("dup_edges_merged", 0)
    ))

    # Reconciliation check
    total_tire = accepted_count + flagged_count + rejected_count
    tire_in_tkg = accepted_count + flagged_count  # both go to memory
    dup = mem_stats.get("dup_edges_merged", 0)
    console.print(
        f"\n  [dim]Edge reconciliation: {accepted_count} ACCEPTED + {flagged_count} FLAGGED"
        f" = {tire_in_tkg} TIRE writes; {dup} duplicate (src,tgt) merged"
        f" → {mem_stats['edges']} unique TKG edges (including MITRE)[/dim]"
    )

    # Confidence histogram — bins on RAW c_NLP (tri.c_nlp is now raw)
    if conf_stats and all_confs:
        ca = np.array(all_confs)
        console.print("\n  [bold]c_NLP distribution (raw, before threshold weighting):[/bold]")
        buckets = [(0.0, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.01)]
        labels  = ["<0.50 REJECTED", "0.50-0.70 FLAGGED",
                   "0.70-0.90 ACCEPT-partial", ">=0.90 ACCEPT-full"]
        for (lo, hi), lbl in zip(buckets, labels):
            count = int(((ca >= lo) & (ca < hi)).sum())
            bar   = "█" * min(30, int(30 * count / max(len(ca), 1)))
            pct   = 100 * count / max(len(ca), 1)
            console.print(f"  [{lbl:24s}] {bar:<30s}  {pct:5.1f}%  ({count:,})")

    # REJECTED spot-check
    if rejected_samples:
        console.print(f"\n  [bold red]REJECTED triplet spot-check[/bold red]"
                      f" (raw c_NLP below 0.50 — excluded from TKG):")
        from rich.table import Table as RichTable
        rt = RichTable(show_header=True, header_style="bold", border_style="dim")
        rt.add_column("Subject",  max_width=22)
        rt.add_column("Relation", max_width=18)
        rt.add_column("Object",   max_width=22)
        rt.add_column("c_NLP",    justify="right")
        rt.add_column("Snippet",  max_width=40)
        for tri in rejected_samples:
            rt.add_row(
                tri.subject[:22],
                tri.relation,
                tri.object_[:22],
                f"{tri.c_nlp:.3f}",
                (tri.source_text or "")[:40].replace("\n", " ")
            )
        console.print(rt)

    # ── Step 7: Export PyG (optional) ────────────────────────────────────────
    if args.export_pyg:
        console.print("\n[bold yellow]⟳ Exporting PyG graph for Phase 2 TGN...[/bold yellow]")
        pyg_data = tkg.to_pyg()
        if pyg_data is not None:
            import torch
            out_path = os.path.join(args.output_dir, "tkg_pyg_snapshot.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(pyg_data, out_path)
            console.print(f"  [green]✓[/green] PyG snapshot saved: {out_path}")
            if hasattr(pyg_data, "edge_index"):
                console.print(
                    f"  Nodes: {pyg_data.num_nodes}  "
                    f"Edges: {pyg_data.edge_index.shape[1]}  "
                    f"Edge features: {pyg_data.edge_attr.shape}"
                )

    # ── Step 8: Subgraph Demo ─────────────────────────────────────────────────
    console.print("\n[bold yellow]⟳ Demo: Multi-hop influence subgraph query...[/bold yellow]")
    # Take first known node for demo
    if known_nodes:
        demo_node = next(iter(known_nodes))
        sg = tkg.get_subgraph_for_phase2([demo_node], hops=2)
        console.print(
            f"  Query node: [bold]{demo_node}[/bold]\n"
            f"  Subgraph: [green]{sg.number_of_nodes()} nodes, "
            f"{sg.number_of_edges()} edges[/green] within 2 hops"
        )

    tkg.shutdown()
    console.print("\n[bold green]✓ Phase 1 pipeline complete![/bold green]")
    console.print(f"  Output artifacts: [bold]{os.path.abspath(args.output_dir)}/[/bold]\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Streaming Ingestion Pipeline")
    parser.add_argument("--samples",    type=int,  default=50,
                        help="Number of TIRE samples to simulate (default: 50)")
    parser.add_argument("--seed-mitre", action="store_true",
                        help="Pre-seed MITRE ATT&CK before streaming")
    parser.add_argument("--export-pyg", action="store_true",
                        help="Export final graph as PyG .pt file for Phase 2")
    parser.add_argument("--neo4j-uri",  type=str,  default=None,
                        help="Neo4j URI (e.g. bolt://localhost:7687); uses JSONL if omitted")
    parser.add_argument("--output-dir", type=str,  default=OUTPUT_DIR,
                        help="Directory to store output artifacts (default: phase1_output)")
    args = parser.parse_args()
    run_pipeline(args)
