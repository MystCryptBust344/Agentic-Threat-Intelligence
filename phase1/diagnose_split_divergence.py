"""
diagnose_split_divergence.py
Finds the EXACT line causing the 43,784 vs 43,996 instance count divergence
between check_entity_leakage.py and train_relation_extractor.py.
Also checks MITRE edge count discrepancy and Unknown->Unknown UUID labeling.
"""
import json
import sys
import os
from collections import defaultdict, Counter

TIRE_PATH  = "../TIRE/dnrti_aug_stix2_je.json"
MITRE_PATH = "../enterprise-attack.json"

# ============================================================
# DIAGNOSTIC 1: Exact total instance count from both parsers
# ============================================================
print("=" * 65)
print("DIAGNOSTIC 1: Instance Count Divergence Root Cause")
print("=" * 65)

with open(TIRE_PATH, encoding="utf-8") as f:
    data = json.load(f)

# Method A: train_relation_extractor.py (groups by text first, then expands)
text_to_instances = defaultdict(list)
skipped_A = 0
for item in data:
    text = item.get("text", "")
    ents = item.get("entities", [])
    for rel in item.get("relations", []):
        rt, i1, i2 = rel[0], rel[1], rel[2]
        if i1 >= len(ents) or i2 >= len(ents):
            skipped_A += 1
            continue
        e1 = ents[i1]
        e2 = ents[i2]
        e1_type = e1[3] if len(e1) > 3 else "UNK"
        e2_type = e2[3] if len(e2) > 3 else "UNK"
        text_to_instances[text].append((e1_type, e2_type, rt))
total_A = sum(len(v) for v in text_to_instances.values())

print(f"\n  Method A (train_relation_extractor.py style):")
print(f"    Unique texts           : {len(text_to_instances):,}")
print(f"    Total instances        : {total_A:,}")
print(f"    Skipped (OOB index)    : {skipped_A}")

# Method B: check_entity_leakage.py (appends each row individually)
all_rows_B = []
skipped_B = 0
for d in data:
    text = d.get("text", "")
    ents = d.get("entities", [])
    for rel in d.get("relations", []):
        rt, i1, i2 = rel[0], rel[1], rel[2]
        if i1 >= len(ents) or i2 >= len(ents):
            skipped_B += 1
            continue
        e1 = ents[i1]
        e2 = ents[i2]
        e1_type = e1[3] if len(e1) > 3 else "UNK"
        e2_type = e2[3] if len(e2) > 3 else "UNK"
        all_rows_B.append({"text": text, "subj_type": e1_type, "obj_type": e2_type, "relation": rt})
total_B = len(all_rows_B)
unique_texts_B = len(set(r["text"] for r in all_rows_B))

print(f"\n  Method B (check_entity_leakage.py style):")
print(f"    Unique texts           : {unique_texts_B:,}")
print(f"    Total instances        : {total_B:,}")
print(f"    Skipped (OOB index)    : {skipped_B}")

print(f"\n  Delta A - B             : {total_A - total_B:+,}")
print(f"  Delta skipped A - B     : {skipped_A - skipped_B:+,}")

if total_A == total_B:
    print("\n  ✓ Both parsers produce IDENTICAL total instance counts.")
    print("  -> Count difference in splits is purely from RNG (now fixed).")
else:
    print(f"\n  ✗ Parsers produce DIFFERENT totals — finding divergence...")
    # Build comparable sets from both methods
    set_A = set()
    for text, instances in text_to_instances.items():
        for (e1t, e2t, rt) in instances:
            set_A.add((text[:80], e1t, e2t, rt))
    set_B = set()
    for r in all_rows_B:
        set_B.add((r["text"][:80], r["subj_type"], r["obj_type"], r["relation"]))
    only_in_A = set_A - set_B
    only_in_B = set_B - set_A
    print(f"  Instances only in A    : {len(only_in_A)}")
    print(f"  Instances only in B    : {len(only_in_B)}")
    if only_in_A:
        print("  Sample only-in-A:", list(only_in_A)[:3])
    if only_in_B:
        print("  Sample only-in-B:", list(only_in_B)[:3])

# ============================================================
# DIAGNOSTIC 2: MITRE edge count — 14,357 seeded vs 20,171 in JSONL
# ============================================================
print("\n" + "=" * 65)
print("DIAGNOSTIC 2: MITRE Record/Edge Count (23,265 vs 20,171 vs 14,357)")
print("=" * 65)

if not os.path.exists(MITRE_PATH):
    print("  enterprise-attack.json not found — skipping.")
else:
    with open(MITRE_PATH, encoding="utf-8") as f:
        mitre_data = json.load(f)
    objects = mitre_data.get("objects", [])

    MITRE_TYPE_MAP = {
        "attack-pattern": "Attack-Pattern",
        "malware": "Malware",
        "tool": "Tool",
        "intrusion-set": "Threat-Actor",
        "course-of-action": "Course-of-Action",
        "campaign": "Campaign",
        "identity": "Identity",
        "vulnerability": "Vulnerability",
    }
    MITRE_REL_MAP = {
        "uses": "uses",
        "mitigates": "mitigates",
        "attributed-to": "attributed-to",
        "subtechnique-of": "subtechnique-of",
        "detects": "detects",
    }

    all_types = Counter(obj.get("type", "unknown") for obj in objects)
    mapped_nodes = {obj["id"] for obj in objects if MITRE_TYPE_MAP.get(obj.get("type"))}
    total_objects = len(objects)
    node_objects = len(mapped_nodes)

    relationships = [obj for obj in objects if obj.get("type") == "relationship"]
    mapped_rels = [r for r in relationships if MITRE_REL_MAP.get(r.get("relationship_type", ""))]
    valid_rels = [r for r in mapped_rels
                  if r.get("source_ref", "") in mapped_nodes
                  and r.get("target_ref", "") in mapped_nodes]

    print(f"\n  Total STIX objects in file   : {total_objects:,}")
    print(f"  Objects with mapped type     : {node_objects:,}  (these become TKG nodes)")
    print(f"  Total relationship objects   : {len(relationships):,}")
    print(f"  With mapped rel_type         : {len(mapped_rels):,}")
    print(f"  Both src+tgt in mapped nodes : {len(valid_rels):,}  <- these become TKG edges (14,357)")
    skipped_unmapped_type = len(relationships) - len(mapped_rels)
    skipped_unmapped_nodes = len(mapped_rels) - len(valid_rels)
    print(f"\n  Skipped (unmapped rel type)  : {skipped_unmapped_type:,}")
    print(f"  Skipped (src or tgt not node): {skipped_unmapped_nodes:,}")
    print(f"\n  '23,265 records' breakdown:")
    print(f"    NODE records (seeded)      : {node_objects:,}")
    print(f"    EDGE records (seeded)      : {len(valid_rels):,}")
    print(f"    Total = 23,265?            : {node_objects + len(valid_rels):,} {'✓' if node_objects + len(valid_rels) == 23265 else 'CHECK'}")
    print(f"\n  '20,171 MITRE edges in JSONL' vs '{len(valid_rels):,} seeded':")
    delta = 20171 - len(valid_rels)
    print(f"    Delta                      : {delta:+,}")
    print(f"    Explanation: TIRE INGESTION can also write edges with source_name='mitre-attack'?")
    print(f"    -> Need to check source_name distribution in JSONL directly.")

# ============================================================
# DIAGNOSTIC 3: Unknown->Unknown UUID node verification
# ============================================================
print("\n" + "=" * 65)
print("DIAGNOSTIC 3: Unknown->Unknown UUID Nodes — STIX type verification")
print("=" * 65)

MUTATIONS_FILE = "phase1_output/tkg_mutations.jsonl"
if not os.path.exists(MUTATIONS_FILE):
    print("  tkg_mutations.jsonl not found.")
else:
    # Collect node_id -> type info from both the JSONL and the MITRE JSON
    stix_type_map = {}
    if os.path.exists(MITRE_PATH):
        with open(MITRE_PATH, encoding="utf-8") as f:
            mitre_data = json.load(f)
        for obj in mitre_data.get("objects", []):
            stix_type_map[obj.get("id", "")] = obj.get("type", "unknown")

    # Find Unknown->Unknown edges (those with no :: in source_id or target_id)
    unknown_src_ids = set()
    unknown_tgt_ids = set()
    sample_edges = []
    with open(MUTATIONS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("type") != "EDGE":
                continue
            src_id = rec.get("source_id", "")
            tgt_id = rec.get("target_id", "")
            if "::" not in src_id and "::" not in tgt_id:
                unknown_src_ids.add(src_id)
                unknown_tgt_ids.add(tgt_id)
                if len(sample_edges) < 10:
                    sample_edges.append(rec)

    print(f"\n  Edges with no '::' in BOTH src AND tgt: {len(sample_edges)} samples shown (of many)")
    print(f"  Unique src node IDs (UUID format)     : {len(unknown_src_ids):,}")
    print(f"  Unique tgt node IDs (UUID format)     : {len(unknown_tgt_ids):,}")
    print(f"\n  Sample 10 edges — STIX type lookup:")
    print(f"  {'source_id (truncated)':<45} {'src_stix_type':<20} {'relation':<20} {'tgt_stix_type'}")
    print("  " + "-" * 110)
    for e in sample_edges:
        src = e["source_id"]
        tgt = e["target_id"]
        src_type = stix_type_map.get(src, "NOT_IN_MITRE")
        tgt_type = stix_type_map.get(tgt, "NOT_IN_MITRE")
        rel = e.get("relation", "?")
        print(f"  {src[:43]:<45} {src_type:<20} {rel:<20} {tgt_type}")

    # Show the actual STIX type distribution for UUID nodes
    src_types = Counter(stix_type_map.get(nid, "NOT_FOUND") for nid in unknown_src_ids)
    tgt_types = Counter(stix_type_map.get(nid, "NOT_FOUND") for nid in unknown_tgt_ids)
    print(f"\n  STIX type distribution of Unknown src nodes:")
    for t, c in src_types.most_common():
        print(f"    {t:<30} : {c:,}")
    print(f"\n  STIX type distribution of Unknown tgt nodes:")
    for t, c in tgt_types.most_common():
        print(f"    {t:<30} : {c:,}")
