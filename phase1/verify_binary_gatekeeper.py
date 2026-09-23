"""
verify_binary_gatekeeper.py

Scientific Verification Script:
1. Exact relation class distribution (positive relations vs noRelation) in dataset & production graph.
2. Checks whether type pairs map strictly to positive relations or if (type_pair, noRelation) co-occurs with (type_pair, positive_relation).
3. Evaluates 4-way feature ablation on the BINARY TASK: has_relation (1) vs noRelation (0).

Usage:
    python -X utf8 verify_binary_gatekeeper.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

DATA_PATH = "../TIRE/dnrti_aug_stix2_je.json"
RANDOM_STATE = 42


def inspect_dataset_distributions():
    print("=" * 75)
    print("  1. Dataset Class Distribution: Positive Relations vs. noRelation")
    print("=" * 75)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_rels = Counter()
    pair_to_positive = defaultdict(Counter)
    pair_to_all = defaultdict(Counter)

    total_instances = 0
    no_relation_count = 0
    positive_relation_count = 0

    for d in data:
        ents = d.get("entities", [])
        for rel in d.get("relations", []):
            rt, i1, i2 = rel[0], rel[1], rel[2]
            if i1 >= len(ents) or i2 >= len(ents):
                continue
            e1_t, e2_t = ents[i1][3], ents[i2][3]
            type_pair = f"{e1_t} -> {e2_t}"

            total_instances += 1
            all_rels[rt] += 1
            pair_to_all[type_pair][rt] += 1

            if rt == "noRelation":
                no_relation_count += 1
            else:
                positive_relation_count += 1
                pair_to_positive[type_pair][rt] += 1

    print(f"\n  Total Triplet Instances : {total_instances:,}")
    print(f"  Positive Relations N   : {positive_relation_count:,} ({100*positive_relation_count/total_instances:.2f}%)")
    print(f"  noRelation N           : {no_relation_count:,} ({100*no_relation_count/total_instances:.2f}%)")

    print(f"\n  Top 5 Relations:")
    for rel_name, count in all_rels.most_common(5):
        print(f"    {rel_name:<20} : {count:>6,} ({100*count/total_instances:.2f}%)")

    # Check co-occurrence of positive relations and noRelation on the SAME type pair
    cooccurrence_pairs = {}
    for pair, rel_counts in pair_to_all.items():
        has_pos = any(r != "noRelation" for r in rel_counts)
        has_neg = "noRelation" in rel_counts
        if has_pos and has_neg:
            cooccurrence_pairs[pair] = dict(rel_counts)

    print("\n" + "─" * 75)
    print(f"  Type Pair Binary Ambiguity Analysis:")
    print(f"    Total Distinct Type Pairs               : {len(pair_to_all)}")
    print(f"    Type Pairs with ONLY positive rels      : {sum(1 for p, c in pair_to_all.items() if 'noRelation' not in c)}")
    print(f"    Type Pairs with ONLY noRelation         : {sum(1 for p, c in pair_to_all.items() if len(c)==1 and 'noRelation' in c)}")
    print(f"    Type Pairs with BOTH positive & noRel   : {len(cooccurrence_pairs)}")
    print("─" * 75)

    if cooccurrence_pairs:
        print("\n  Sample Type Pairs exhibiting Binary Ambiguity (Positive vs noRelation):")
        print(f"  {'Type Pair':<30} {'Total N':>8}  Breakdown (noRelation vs Positive)")
        print("  " + "-" * 70)
        for pair, rel_dict in sorted(cooccurrence_pairs.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]:
            tot = sum(rel_dict.values())
            print(f"  {pair:<30} {tot:>8,}  {rel_dict}")

    return data, pair_to_all, cooccurrence_pairs


def evaluate_binary_gatekeeper(data):
    print("\n" + "=" * 75)
    print("  2. Binary Task Evaluation: has_relation (1) vs. noRelation (0)")
    print("=" * 75)

    # Parse all instances first — only texts with >= 1 valid relation appear
    # (matches train_relation_extractor.py's list(text_to_instances.keys()))
    from collections import defaultdict as _dd
    text_to_rows = _dd(list)
    for d in data:
        text = d.get("text", "")
        ents = d.get("entities", [])
        for rel in d.get("relations", []):
            rt, i1, i2 = rel[0], rel[1], rel[2]
            if i1 >= len(ents) or i2 >= len(ents):
                continue
            e1, e2 = ents[i1], ents[i2]
            text_to_rows[text].append({
                "text": text,
                "subj": e1[2] if len(e1) > 2 else "",
                "subj_type": e1[3] if len(e1) > 3 else "UNK",
                "obj": e2[2] if len(e2) > 2 else "",
                "obj_type": e2[3] if len(e2) > 3 else "UNK",
                "binary_label": 1 if rt != "noRelation" else 0,
                "full_relation": rt
            })

    # Shuffle only texts with >= 1 instance (insertion order = canonical)
    unique_texts = list(text_to_rows.keys())   # 5,398 texts, insertion order
    rng = np.random.default_rng(RANDOM_STATE)
    unique_texts = rng.permutation(unique_texts).tolist()

    n = len(unique_texts)
    n_val   = int(n * 0.15)
    n_test  = int(n * 0.15)
    n_train = n - n_val - n_test   # train gets rounding remainder

    train_texts = set(unique_texts[:n_train])
    val_texts   = set(unique_texts[n_train:n_train + n_val])

    train_rows, val_rows = [], []
    for text, rows in text_to_rows.items():
        if text in train_texts:
            train_rows.extend(rows)
        elif text in val_texts:
            val_rows.extend(rows)

    print(f"  Binary Train Instances : {len(train_rows):,}")
    print(f"  Binary Val Instances   : {len(val_rows):,}")
    # ── Regression Guard ─────────────────────────────────────────────────────
    # Must match train_relation_extractor.py's canonical split (fast_extractor.pkl).
    # If this fires, check: seed=42, default_rng, valid-text-only scope, n_train=n-n_val-n_test.
    # All three root causes documented in git history (commit a28b711).
    assert len(train_rows) == 43_996, (
        f"Split regression: train={len(train_rows):,} != 43,996. "
        f"Split must match train_relation_extractor.load_tire_data_dedup()"
    )
    assert len(val_rows) == 8_519, (
        f"Split regression: val={len(val_rows):,} != 8,519. "
        f"Split must match train_relation_extractor.load_tire_data_dedup()"
    )
    # ─────────────────────────────────────────────────────────────────────────

    y_train = [r["binary_label"] for r in train_rows]
    y_val   = [r["binary_label"] for r in val_rows]

    majority_label = Counter(y_train).most_common(1)[0][0]
    majority_acc   = accuracy_score(y_val, [majority_label] * len(y_val))

    print(f"  Val Set Majority Class ('{'has_relation' if majority_label==1 else 'noRelation'}'): {majority_acc:.4f} ({100*majority_acc:.1f}%)")

    # Feature Builders
    def b_baseline(r):
        text = r["text"]
        e1, e1_t, e2, e2_t = r["subj"], r["subj_type"], r["obj"], r["obj_type"]
        marked = text
        if e1 and e1 in marked: marked = marked.replace(e1, f"[E1] {e1} [/E1]", 1)
        if e2 and e2 in marked and e1 != e2: marked = marked.replace(e2, f"[E2] {e2} [/E2]", 1)
        return f"{e1_t} REL {e2_t} | {marked}"

    def b_masked(r):
        text = r["text"]
        e1, e1_t, e2, e2_t = r["subj"], r["subj_type"], r["obj"], r["obj_type"]
        masked = text
        spans = [s for s in {e1, e2} if isinstance(s, str) and s]
        for span in sorted(spans, key=len, reverse=True):
            if span in masked:
                placeholder = "__SUBJ__" if span == e1 else "__OBJ__"
                masked = masked.replace(span, placeholder)
        return f"{e1_t} REL {e2_t} | {masked}"

    def b_types_only(r):
        return f"{r['subj_type']} REL {r['obj_type']}"

    def b_pure_context(r):
        text, e1, e2 = r["text"], r["subj"], r["obj"]
        masked = text
        spans = [s for s in {e1, e2} if isinstance(s, str) and s]
        for span in sorted(spans, key=len, reverse=True):
            if span in masked:
                placeholder = "__SUBJ__" if span == e1 else "__OBJ__"
                masked = masked.replace(span, placeholder)
        return masked

    variants = [
        ("1. BASELINE (Types + Text + Context)", b_baseline),
        ("2. MASKED_TEXT (Types + Masked Context)", b_masked),
        ("3. ENTITY_TYPES_ONLY (Type Pair Only)", b_types_only),
        ("4. PURE_CONTEXT_ONLY (Sentence Text Only)", b_pure_context),
    ]

    print("\n" + "─" * 75)
    print("  Binary Task 4-Way Feature Ablation Results:")
    print("─" * 75)

    for vname, vfn in variants:
        X_train = [vfn(r) for r in train_rows]
        X_val   = [vfn(r) for r in val_rows]

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
        f1 = f1_score(y_val, preds, average="binary")

        print(f"  {vname:<45} : Acc={acc:.4f} ({100*acc:.1f}%)"
              f" | Binary F1={f1:.4f} | Delta: {(acc-majority_acc)*100:+.1f} pp")

    print("=" * 75)


def main():
    data, _, _ = inspect_dataset_distributions()
    evaluate_binary_gatekeeper(data)


if __name__ == "__main__":
    main()
