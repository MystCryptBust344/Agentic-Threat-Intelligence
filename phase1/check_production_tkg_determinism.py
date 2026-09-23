"""
check_production_tkg_determinism.py

Production TKG Analysis:
1. Replays tkg_mutations.jsonl through InMemoryGraphStore.add_edge() —
   the SAME merge code path validated by test_phase1.py TEST 1.
   This produces the post-merge, deduplication-consistent graph edge count,
   not raw JSONL event counts.
2. Reports raw event count vs post-merge count vs PyG snapshot (41,646) for
   full reconciliation transparency.
3. Computes (source_type_label, target_type_label) -> relation distribution
   on the post-merge graph and checks type-pair determinism.

Merge Key Reality (tkg_store.py InMemoryGraphStore.add_edge, lines 324-343):
  NetworkX DiGraph stores ONE edge per directed (u, v) pair.
  The merge key is (source_id, target_id) ONLY.
  A second add_edge() call with the same (u, v) but different relation OVERWRITES the first.
  This matches the behavior the PyG snapshot is built from.

IMPORTANT — Run this on a FRESH mutation log (after clearing tkg_mutations.jsonl
and running ingestion_pipeline.py once). The JSONL log is append-only across runs.
If the log has accumulated across multiple sessions, raw event count will be inflated
but post-merge replay will still produce the correct graph (InMemoryGraphStore deduplicates
correctly regardless of how many times events are replayed).

Usage:
    python -X utf8 check_production_tkg_determinism.py
"""

import json
import os
import sys
from collections import Counter, defaultdict

MUTATIONS_FILE = "phase1_output/tkg_mutations.jsonl"
PYG_REFERENCE_EDGES = 41_646   # expected post-merge edge count from PyG snapshot
PYG_REFERENCE_NODES =  9_417   # expected unique node count from PyG snapshot


def run_production_analysis():
    if not os.path.exists(MUTATIONS_FILE):
        print(f"Error: {MUTATIONS_FILE} not found. Run ingestion_pipeline.py first.")
        sys.exit(1)

    print("=" * 75)
    print("  Production TKG Schema Determinism Analysis")
    print("  (Replayed through InMemoryGraphStore — graph-consistent post-merge counts)")
    print("=" * 75)

    # ── Step 1: Count raw JSONL events ───────────────────────────────────────
    raw_node_events = 0
    raw_edge_events = 0
    with open(MUTATIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "NODE":
                raw_node_events += 1
            elif rec.get("type") == "EDGE":
                raw_edge_events += 1

    print(f"\n  [Step 1] Raw JSONL Mutation Log:")
    print(f"    NODE events in log : {raw_node_events:,}")
    print(f"    EDGE events in log : {raw_edge_events:,}  ← pre-merge (includes all duplicate writes)")

    # ── Step 2: Replay through InMemoryGraphStore (real merge code path) ─────
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tkg_store import InMemoryGraphStore, TKGNode, TKGEdge

    memory = InMemoryGraphStore()

    with open(MUTATIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "NODE":
                try:
                    n = TKGNode(**{k: v for k, v in rec.items() if k != "type"})
                    memory.add_node(n)
                except Exception:
                    pass
            elif rec.get("type") == "EDGE":
                try:
                    e = TKGEdge(**{k: v for k, v in rec.items() if k != "type"})
                    memory.add_edge(e)
                except Exception:
                    pass

    post_merge_edges = list(memory._graph.edges(data=True))
    post_merge_count = len(post_merge_edges)
    post_merge_nodes = memory._graph.number_of_nodes()
    dup_merges = memory._dup_count

    print(f"\n  [Step 2] InMemoryGraphStore Replay Results:")
    print(f"    Duplicate (u,v) merges applied : {dup_merges:,}")
    print(f"    Post-merge unique edges        : {post_merge_count:,}  ← graph-consistent count")
    print(f"    PyG snapshot reference (edges) : {PYG_REFERENCE_EDGES:,}")
    edge_delta = post_merge_count - PYG_REFERENCE_EDGES
    edge_match = "✓ EXACT MATCH" if edge_delta == 0 else f"✗ MISMATCH (delta={edge_delta:+,})"
    print(f"    Edge reconciliation            : {edge_match}")
    print(f"    Post-merge unique nodes        : {post_merge_nodes:,}")
    print(f"    PyG snapshot reference (nodes) : {PYG_REFERENCE_NODES:,}")
    node_delta = post_merge_nodes - PYG_REFERENCE_NODES
    node_match = "✓ EXACT MATCH" if node_delta == 0 else f"✗ MISMATCH (delta={node_delta:+,})"
    print(f"    Node reconciliation            : {node_match}")

    # ── Step 3: Source breakdown ──────────────────────────────────────────────
    source_counts = Counter()
    for u, v, d in post_merge_edges:
        source_counts[d.get("source_name", "unknown")] += 1

    print(f"\n  Edge Source Breakdown (post-merge):")
    for src_name, count in source_counts.most_common():
        print(f"    {src_name:<20} : {count:>7,} edges ({100*count/post_merge_count:.1f}%)")

    # ── Step 4: Type-pair determinism on the post-merge graph ─────────────────
    print(f"\n  [Step 3] Type-pair determinism on {post_merge_count:,} post-merge edges...")

    pair_to_rels = defaultdict(Counter)
    for u, v, d in post_merge_edges:
        src_label = u.split("::")[0] if "::" in u else "Unknown"
        tgt_label = v.split("::")[0] if "::" in v else "Unknown"
        rel = d.get("relation", "unknown")
        pair_to_rels[f"{src_label} -> {tgt_label}"][rel] += 1

    deterministic_pairs = {}
    ambiguous_pairs = {}

    for pair, rel_counts in sorted(pair_to_rels.items(), key=lambda x: sum(x[1].values()), reverse=True):
        total_n = sum(rel_counts.values())
        top_rel, top_count = rel_counts.most_common(1)[0]
        top_pct = 100.0 * top_count / total_n
        num_unique = len(rel_counts)
        if num_unique == 1:
            deterministic_pairs[pair] = (total_n, top_rel, top_pct)
        else:
            ambiguous_pairs[pair] = (total_n, num_unique, dict(rel_counts), top_rel, top_pct)

    n_det = sum(v[0] for v in deterministic_pairs.values())
    n_amb = sum(v[0] for v in ambiguous_pairs.values())

    print(f"\n{'=' * 75}")
    print(f"  PRODUCTION TKG DETERMINISM SUMMARY (Post-Merge Graph)")
    print(f"{'=' * 75}")
    print(f"  Total Post-Merge Unique Edges          : {post_merge_count:,}")
    print(f"  Distinct Type Pairs                    : {len(pair_to_rels)}")
    print(f"  Deterministic Pairs (1 relation/pair)  : {len(deterministic_pairs)}"
          f"  ({n_det:,} edges, {100*n_det/post_merge_count:.1f}%)")
    print(f"  Ambiguous Pairs (>1 relation/pair)     : {len(ambiguous_pairs)}"
          f"  ({n_amb:,} edges, {100*n_amb/post_merge_count:.1f}%)")
    print(f"{'=' * 75}")

    if ambiguous_pairs:
        print(f"\n  Ambiguous Type Pairs (post-merge graph):")
        print(f"  {'Type Pair':<30} {'Edges':>7}  Relation Distribution")
        print("  " + "-" * 70)
        for p, (tot, num_u, rel_dict, top_r, top_p) in sorted(
                ambiguous_pairs.items(), key=lambda x: x[1][0], reverse=True)[:15]:
            print(f"  {p:<30} {tot:>7,}  {rel_dict}")
    else:
        print("\n  ✓ FULLY DETERMINISTIC: Every type pair in the post-merge graph maps to exactly 1 relation.")

    return ambiguous_pairs


if __name__ == "__main__":
    run_production_analysis()
