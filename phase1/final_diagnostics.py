"""
final_diagnostics.py
====================
Runs the remaining four rigorous diagnostics for Phase 1:
1. Threshold boundary synchronization between extraction and storage.
2. Corroboration distribution check (sparsity of obs_count).
3. Entity-resolution (surface-form deduplication) quality check.
4. NER garbage-rate scan at full-corpus scale.
"""

import json
import re
import sys
from collections import defaultdict

# Phase 1 imports
from triplet_extractor import TripletExtractor, extract_entities_heuristic
from tkg_store import DualLayerTKG

def test_threshold_sync():
    print("============================================================")
    print("  1. Threshold Boundary Synchronization")
    print("============================================================")
    mismatches = 0
    for i in range(101):
        c = i / 100.0
        
        # TripletExtractor returns (weight, status)
        w_ext, s_ext = TripletExtractor._apply_threshold(c)
        
        # DualLayerTKG returns weight (or None). 
        w_tkg = DualLayerTKG._apply_threshold(c)
        
        if w_ext != w_tkg:
            print(f"  MISMATCH at c={c:.2f}: Extractor={w_ext}, TKGStore={w_tkg}")
            mismatches += 1
            
    if mismatches == 0:
        print("  [PASS] _apply_threshold identical across all c in [0.0, 1.0].")
        print("         The two independent implementations are fully synced.")
    else:
        print(f"  [FAIL] {mismatches} mismatches found between the two policies!")
    print()

def check_corroboration_dist(log_path="phase1_output/tkg_mutations.jsonl"):
    print("============================================================")
    print("  2. Corroboration Distribution Check")
    print("============================================================")
    pair_counts = defaultdict(int)
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") == "EDGE":
                    u = rec.get("source_id")
                    v = rec.get("target_id")
                    if u and v:
                        pair_counts[(u, v)] += 1
                        
        hist = {"1": 0, "2": 0, "3": 0, "4-10": 0, ">10": 0}
        total_edges = len(pair_counts)
        
        if total_edges == 0:
            print("  No edges found in mutations log.")
            return

        for count in pair_counts.values():
            if count == 1: hist["1"] += 1
            elif count == 2: hist["2"] += 1
            elif count == 3: hist["3"] += 1
            elif count <= 10: hist["4-10"] += 1
            else: hist[">10"] += 1
            
        print(f"  Total post-merge node pairs: {total_edges}")
        for k, v in hist.items():
            print(f"    obs_count {k:<5} : {v:>6} edges ({v/total_edges*100:.1f}%)")
            
        multi = total_edges - hist["1"]
        fraction = multi / total_edges * 100
        print(f"\n  Fraction with obs_count > 1: {fraction:.2f}%")
        
        if fraction < 5.0:
            print("  [WARNING] Corroboration signal is highly sparse!")
            print("            Most edges rely purely on extraction confidence, not multi-report frequency.")
    except FileNotFoundError:
        print(f"  [ERROR] '{log_path}' not found. Run ingestion_pipeline.py first.")
    print()

def check_entity_resolution(nodes_path="phase1_output/tkg_node_index.json"):
    print("============================================================")
    print("  3. Entity-Resolution Dedup Quality")
    print("============================================================")
    try:
        with open(nodes_path, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
            
        norm_groups = defaultdict(set)
        for node_id in nodes.keys():
            # Format is usually "TYPE::Raw Text"
            raw_label = node_id.split("::", 1)[1] if "::" in node_id else node_id
            
            # Normalize: lowercase, strip punctuation, collapse internal whitespace
            norm = re.sub(r'[^\w\s]', '', raw_label).lower()
            norm = re.sub(r'\s+', ' ', norm).strip()
            
            if norm:
                norm_groups[norm].add(node_id)
                
        multi_groups = {k: v for k, v in norm_groups.items() if len(v) > 1}
        print(f"  Total raw nodes              : {len(nodes)}")
        print(f"  Total normalized concepts    : {len(norm_groups)}")
        print(f"  Fragmented entities (groups) : {len(multi_groups)}")
        
        if multi_groups:
            print("\n  Top 10 most fragmented entities (by variant count):")
            sorted_groups = sorted(multi_groups.items(), key=lambda x: len(x[1]), reverse=True)
            for k, v in sorted_groups[:10]:
                print(f"    '{k}' ({len(v)} variants)")
                for var in list(v)[:3]:
                    print(f"       - {var}")
                if len(v) > 3:
                    print(f"       - ... and {len(v)-3} more")
        else:
            print("  [PASS] No surface-form fragmentation detected.")
            
    except FileNotFoundError:
        print(f"  [ERROR] '{nodes_path}' not found. Run ingestion_pipeline.py first.")
    print()

def check_ner_fp_rate(dataset_path="../TIRE/dnrti_aug_stix2_je.json"):
    print("============================================================")
    print("  4. NER False-Positive Scan (Heuristic)")
    print("============================================================")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            
        total_ents = 0
        flagged = []
        
        is_digit = re.compile(r'^\d+$')
        is_punct = re.compile(r'^[^\w\s]+$')
        
        # A conservative list of highly common english words that should never be entities alone
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
            "by", "from", "of", "this", "that", "it", "is", "was", "are", "were", "be", 
            "been", "has", "have", "had", "do", "does", "did", "will", "would", "shall", 
            "should", "can", "could", "may", "might", "must", "not", "no", "yes"
        }
        
        for sample in samples:
            # Depending on dataset format, the sentence could be in 'token' or 'sentence' or 'text'
            text = sample.get("token") or sample.get("sentence") or sample.get("text") or ""
            if isinstance(text, list):
                text = " ".join(text)
                
            if not text: 
                continue
            
            ents = extract_entities_heuristic(text)
            for e_text, e_label, _ in ents:
                total_ents += 1
                clean = e_text.lower().strip()
                
                # Flag if pure digits, pure punctuation, or a single common stopword
                if is_digit.match(clean) or is_punct.match(clean) or clean in common_words:
                    flagged.append((e_text, e_label, text[:80] + "..."))
                    
        print(f"  Total CTI samples processed  : {len(samples)}")
        print(f"  Total entities extracted     : {total_ents}")
        print(f"  Entities flagged as garbage  : {len(flagged)}")
        
        if total_ents > 0:
            fpr = (len(flagged) / total_ents) * 100
            print(f"  Rough False-Positive Rate    : {fpr:.2f}%")
            
            if flagged:
                print("\n  Examples of extracted garbage:")
                for e_text, e_label, ctx in flagged[:10]:
                    print(f"    - '{e_text}' [{e_label}]  (Context: {ctx})")
                    
    except FileNotFoundError:
        print(f"  [ERROR] Dataset '{dataset_path}' not found.")
    except Exception as e:
        print(f"  [ERROR] Failed to run NER scan: {e}")
    print()

if __name__ == "__main__":
    test_threshold_sync()
    check_corroboration_dist()
    check_entity_resolution()
    check_ner_fp_rate()
