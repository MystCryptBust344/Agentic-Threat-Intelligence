"""
analyze_schema_ambiguity.py

Deep Scientific Analysis of STIX2 / TIRE Dataset Schema:
1. Computes (subject_type, object_type) -> relation distribution & entropy.
2. Identifies deterministic type pairs (100% single relation) vs ambiguous type pairs (>1 relation).
3. Evaluates relation extraction model variants on the AMBIGUOUS SUBSET ONLY.

Usage:
    python -X utf8 analyze_schema_ambiguity.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

DATA_PATH = "../TIRE/dnrti_aug_stix2_je.json"
RANDOM_STATE = 42


def load_raw_triplets():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    triplets = []
    for d in data:
        text = d.get("text", "")
        ents = d.get("entities", [])
        for rel in d.get("relations", []):
            rt, i1, i2 = rel[0], rel[1], rel[2]
            if i1 >= len(ents) or i2 >= len(ents):
                continue
            e1 = ents[i1]
            e2 = ents[i2]
            subj_str  = e1[2] if len(e1) > 2 else ""
            subj_type = e1[3] if len(e1) > 3 else "UNK"
            obj_str   = e2[2] if len(e2) > 2 else ""
            obj_type  = e2[3] if len(e2) > 3 else "UNK"

            triplets.append({
                "text": text,
                "subj": subj_str,
                "subj_type": subj_type,
                "relation": rt,
                "obj": obj_str,
                "obj_type": obj_type,
                "type_pair": f"{subj_type} -> {obj_type}"
            })
    return data, triplets


def analyze_type_pair_distribution(triplets):
    print("\n" + "=" * 75)
    print("  1. (Subject Type, Object Type) -> Relation Schema Distribution")
    print("=" * 75)

    pair_to_rels = defaultdict(Counter)
    for t in triplets:
        pair_to_rels[t["type_pair"]][t["relation"]] += 1

    deterministic_pairs = {}
    ambiguous_pairs = {}

    print(f"\n{'Type Pair (Subj -> Obj)':<30} {'Total N':>8} {'Unique Rels':>12}  Top Relation (Dominance %)")
    print("-" * 75)

    for pair, rel_counts in sorted(pair_to_rels.items(), key=lambda x: sum(x[1].values()), reverse=True):
        total_n = sum(rel_counts.values())
        top_rel, top_count = rel_counts.most_common(1)[0]
        top_pct = 100.0 * top_count / total_n
        num_unique = len(rel_counts)

        if num_unique == 1 or top_pct >= 99.0:
            deterministic_pairs[pair] = (total_n, top_rel, top_pct)
        else:
            ambiguous_pairs[pair] = (total_n, num_unique, dict(rel_counts), top_rel, top_pct)

        flag = " [DETERMINISTIC]" if top_pct >= 99.0 else " [AMBIGUOUS]"
        print(f"{pair:<30} {total_n:>8,} {num_unique:>12d}  {top_rel:<20} ({top_pct:5.1f}%){flag}")

    n_total_instances = len(triplets)
    n_det_instances = sum(sum(pair_to_rels[p].values()) for p in deterministic_pairs)
    n_amb_instances = sum(sum(pair_to_rels[p].values()) for p in ambiguous_pairs)

    print("\n" + "─" * 75)
    print("  Schema Determinism Summary:")
    print(f"    Total Triplet Instances : {n_total_instances:,}")
    print(f"    Deterministic Pairs N   : {n_det_instances:,} ({100*n_det_instances/n_total_instances:.1f}% of dataset)")
    print(f"    Ambiguous Pairs N       : {n_amb_instances:,} ({100*n_amb_instances/n_total_instances:.1f}% of dataset)")
    print("─" * 75)

    return deterministic_pairs, ambiguous_pairs


def evaluate_on_subset(name, train_triplets, val_triplets):
    print(f"\n" + "=" * 75)
    print(f"  Evaluating Model Variants on: {name}")
    print(f"  Train N: {len(train_triplets):,} | Val N: {len(val_triplets):,}")
    print("=" * 75)

    def build_baseline(t):
        e1, e1_t = t["subj"], t["subj_type"]
        e2, e2_t = t["obj"], t["obj_type"]
        marked = t["text"]
        if e1 and e1 in marked: marked = marked.replace(e1, f"[E1] {e1} [/E1]", 1)
        if e2 and e2 in marked and e1 != e2: marked = marked.replace(e2, f"[E2] {e2} [/E2]", 1)
        return f"{e1_t} REL {e2_t} | {marked}"

    def build_masked(t):
        e1, e1_t = t["subj"], t["subj_type"]
        e2, e2_t = t["obj"], t["obj_type"]
        masked = t["text"]
        spans = [s for s in {e1, e2} if isinstance(s, str) and s]
        for span in sorted(spans, key=len, reverse=True):
            if span in masked:
                placeholder = "__SUBJ__" if span == e1 else "__OBJ__"
                masked = masked.replace(span, placeholder)
        return f"{e1_t} REL {e2_t} | {masked}"

    def build_types_only(t):
        return f"{t['subj_type']} REL {t['obj_type']}"

    def build_pure_context(t):
        e1, e2 = t["subj"], t["obj"]
        masked = t["text"]
        spans = [s for s in {e1, e2} if isinstance(s, str) and s]
        for span in sorted(spans, key=len, reverse=True):
            if span in masked:
                placeholder = "__SUBJ__" if span == e1 else "__OBJ__"
                masked = masked.replace(span, placeholder)
        return masked

    feature_fns = [
        ("1. BASELINE (Types + Text + Context)", build_baseline),
        ("2. MASKED_TEXT (Types + Masked Context)", build_masked),
        ("3. ENTITY_TYPES_ONLY (Type Pair Only)", build_types_only),
        ("4. PURE_CONTEXT_ONLY (Sentence Text Only)", build_pure_context),
    ]

    y_train = [t["relation"] for t in train_triplets]
    y_val   = [t["relation"] for t in val_triplets]

    majority_label = Counter(y_train).most_common(1)[0][0]
    majority_acc   = accuracy_score(y_val, [majority_label] * len(y_val))

    print(f"\n  Majority Baseline on {name}: {majority_acc:.4f} ({100*majority_acc:.1f}%)")
    print("-" * 75)

    results = {}
    for fname, ffn in feature_fns:
        X_train = [ffn(t) for t in train_triplets]
        X_val   = [ffn(t) for t in val_triplets]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=60_000,
                                      sublinear_tf=True, strip_accents="unicode")),
            ("norm", Normalizer()),
            ("clf", LogisticRegression(C=5.0, class_weight="balanced", solver="lbfgs",
                                       max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE)),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        acc = accuracy_score(y_val, preds)
        results[fname] = acc
        print(f"  {fname:<45} : {acc:.4f} ({100*acc:.1f}%)  [Delta: {(acc-majority_acc)*100:+.1f} pp]")

    return results, majority_acc


def text_dedup_split(triplets, seed=RANDOM_STATE):
    """
    FIXED: previously used np.random.RandomState + sorted-then-shuffled texts
    + int(n*0.70) rounding (remainder silently dropped to test). That is a
    DIFFERENT split from train_relation_extractor.py's load_tire_data_dedup(),
    which trained fast_extractor.pkl — same class of bug already found and
    fixed in check_entity_leakage.py / verify_binary_gatekeeper.py (RNG API +
    rounding + text-set-scope divergence, git a28b711). This version matches
    the canonical split exactly: default_rng, insertion-order texts before
    permutation, and the rounding remainder assigned to train.

    `triplets` is already scoped to texts with >=1 in-bounds relation instance
    (load_raw_triplets never emits an entry otherwise), so this matches the
    canonical split's text scope without further filtering.
    """
    text_to_triplets = defaultdict(list)
    for t in triplets:
        text_to_triplets[t["text"]].append(t)

    unique_texts = list(text_to_triplets.keys())   # insertion order — must match canonical
    rng = np.random.default_rng(seed)              # was RandomState — different shuffle algorithm
    shuffled = rng.permutation(unique_texts).tolist()

    n = len(shuffled)
    n_val   = int(n * 0.15)
    n_test  = int(n * 0.15)
    n_train = n - n_val - n_test   # remainder -> train (was silently -> test)

    train_texts = set(shuffled[:n_train])
    val_texts   = set(shuffled[n_train:n_train + n_val])
    test_texts  = set(shuffled[n_train + n_val:])

    train = [t for text in train_texts for t in text_to_triplets[text]]
    val   = [t for text in val_texts for t in text_to_triplets[text]]
    test  = [t for text in test_texts for t in text_to_triplets[text]]

    # Regression guard — should match the canonical split's instance counts
    # exactly (43,996 / 8,519) since this dataset includes noRelation
    # instances, same as check_entity_leakage.py's un-filtered scope.
    if len(train) != 43_996 or len(val) != 8_519:
        print(f"  [WARNING] Split sizes (train={len(train):,}, val={len(val):,}) "
              f"do not match the canonical split (43,996 / 8,519). "
              f"If this fires, compare against check_entity_leakage.py's split logic.")

    return train, val, test


def main():
    print("=" * 75)
    print("  STIX2 / DNRTI Schema Ambiguity & NLP Context Evaluation")
    print("=" * 75)

    raw_data, all_triplets = load_raw_triplets()
    det_pairs, amb_pairs = analyze_type_pair_distribution(all_triplets)

    train_all, val_all, _ = text_dedup_split(all_triplets)

    # Filter train & val splits to ambiguous type pairs only
    amb_pair_set = set(amb_pairs.keys())
    train_amb = [t for t in train_all if t["type_pair"] in amb_pair_set]
    val_amb   = [t for t in val_all if t["type_pair"] in amb_pair_set]

    # Evaluate full dataset vs ambiguous subset
    res_full, maj_full = evaluate_on_subset("FULL DATASET", train_all, val_all)
    res_amb, maj_amb   = evaluate_on_subset("AMBIGUOUS SUBSET ONLY", train_amb, val_amb)

    print("\n" + "=" * 75)
    print("  COMPARATIVE SUMMARY: FULL DATASET vs. AMBIGUOUS SUBSET")
    print("=" * 75)
    print(f"{'Feature Variant':<45} {'Full Dataset':>14} {'Ambiguous Subset':>18}")
    print("-" * 75)
    print(f"{'Majority Baseline':<45} {maj_full:>14.1f}% {maj_amb:>18.1f}%")
    for k in res_full:
        print(f"{k:<45} {100*res_full[k]:>13.1f}% {100*res_amb[k]:>17.1f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()