"""
verify_relation_merge.py
========================
Verifies that no distinct relations are lost due to the (source, target) merge key.
Reads the mutation log, groups EDGE events by (source_id, target_id), and counts
if any group contains more than one unique relation value.
"""

import json
from collections import defaultdict
import os

def check_merge_conflicts(log_path="phase1_output/tkg_mutations.jsonl"):
    if not os.path.exists(log_path):
        print(f"ERROR: {log_path} not found. Run ingestion_pipeline.py first.")
        return
        
    pair_relations = defaultdict(set)
    edge_count = 0
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "EDGE":
                edge_count += 1
                src = rec.get("source_id")
                tgt = rec.get("target_id")
                rel = rec.get("relation")
                if src and tgt and rel:
                    pair_relations[(src, tgt)].add(rel)
                    
    multi_rel_pairs = {k: v for k, v in pair_relations.items() if len(v) > 1}
    
    print("============================================================")
    print("  Relation Merge Conflict Verification")
    print("============================================================")
    print(f"Total raw EDGE events processed: {edge_count}")
    print(f"Total distinct (source_id, target_id) pairs: {len(pair_relations)}")
    print(f"Pairs containing multiple distinct relations: {len(multi_rel_pairs)}")
    print("------------------------------------------------------------")
    
    if len(multi_rel_pairs) > 0:
        print("[FAIL] Conflicts detected! The following node pairs have multiple relations:")
        for idx, (pair, rels) in enumerate(list(multi_rel_pairs.items())[:10]):
            print(f"  {pair[0]} -> {pair[1]} : {rels}")
        if len(multi_rel_pairs) > 10:
            print(f"  ... and {len(multi_rel_pairs) - 10} more.")
    else:
        print("[PASS] Zero conflicts detected. The (u, v) merge key discards no relations.")
        print("       Every merged node pair shares the exact same relation type.")

if __name__ == "__main__":
    check_merge_conflicts()
