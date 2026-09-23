"""
check_entity_leakage.py

Diagnostic & Ablation Study:
Disentangling relation extraction performance across 4 feature variants:
  1. BASELINE          : Entity types + Entity text + Sentence context
  2. MASKED_TEXT       : Entity types + Masked text (__SUBJ__/__OBJ__) + Sentence context
  3. ENTITY_TYPES_ONLY : Entity type pair ONLY (e.g. "APT REL MAL")
  4. PURE_CONTEXT_ONLY : Masked text ONLY (NO entity types, NO entity text strings)

Optimization & Convergence:
  - Uses solver='lbfgs' with max_iter=2000 to eliminate ConvergenceWarning.
  - Applies TfidfVectorizer with sublinear_tf=True and strip_accents='unicode'.

Run:
    python -X utf8 check_entity_leakage.py
"""

import json
import os
import sys
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

DATA_PATH = "../TIRE/dnrti_aug_stix2_je.json"
RANDOM_STATE = 42


def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data


def text_dedup_split(data, train_frac=0.70, val_frac=0.15, seed=RANDOM_STATE):
    """
    Sentence-level text-deduplicated split — exactly mirrors
    train_relation_extractor.py's load_tire_data_dedup() structure:

      1. Parse ALL instances first into text_to_instances (insertion order).
         Only texts with >= 1 valid instance appear — matches
         list(text_to_instances.keys()) in train_relation_extractor.py.
      2. Shuffle unique texts with default_rng (same API as canonical split).
      3. Assign rounding remainder to train (n_train = n - n_val - n_test).
      4. Expand each split bucket from its assigned texts.

    This produces Train=43,996 / Val=8,519, identical to fast_extractor.pkl's
    training split.
    """
    from collections import defaultdict as _dd

    # Step 1: parse all instances, group by text (insertion order, valid only)
    text_to_rows = _dd(list)
    for d in data:
        text = d.get("text", "")
        ents = d.get("entities", [])
        for rel in d.get("relations", []):
            rt, i1, i2 = rel[0], rel[1], rel[2]
            if i1 >= len(ents) or i2 >= len(ents):
                continue
            e1 = ents[i1]
            e2 = ents[i2]
            text_to_rows[text].append({
                "text": text,
                "subj": e1[2] if len(e1) > 2 else "",
                "subj_type": e1[3] if len(e1) > 3 else "UNK",
                "relation": rt,
                "obj": e2[2] if len(e2) > 2 else "",
                "obj_type": e2[3] if len(e2) > 3 else "UNK",
            })

    # Step 2: shuffle only texts that produced >= 1 instance (insertion order)
    unique_texts = list(text_to_rows.keys())      # 5,398 texts, insertion order
    rng = np.random.default_rng(seed)             # matches canonical RNG
    shuffled = rng.permutation(unique_texts).tolist()

    # Step 3: compute split sizes — train gets the rounding remainder
    n = len(shuffled)
    n_val  = int(n * val_frac)
    n_test = int(n * (1.0 - train_frac - val_frac))
    n_train = n - n_val - n_test

    texts_train = set(shuffled[:n_train])
    texts_val   = set(shuffled[n_train:n_train + n_val])
    texts_test  = set(shuffled[n_train + n_val:])

    # Step 4: expand into split buckets
    train, val, test = [], [], []
    for text, rows in text_to_rows.items():
        if text in texts_train:
            train.extend(rows)
        elif text in texts_val:
            val.extend(rows)
        else:
            test.extend(rows)

    return train, val, test


# ── Feature Builders ──────────────────────────────────────────────────────────

def build_feature_baseline(row):
    """1. BASELINE: Entity types + Entity text + Sentence context."""
    text = row["text"]
    e1, e1_t = row["subj"], row["subj_type"]
    e2, e2_t = row["obj"], row["obj_type"]
    marked = text
    if e1 and e1 in marked:
        marked = marked.replace(e1, f"[E1] {e1} [/E1]", 1)
    if e2 and e2 in marked and e1 != e2:
        marked = marked.replace(e2, f"[E2] {e2} [/E2]", 1)
    return f"{e1_t} REL {e2_t} | {marked}"


def build_feature_masked(row):
    """2. MASKED_TEXT: Entity types + Masked sentence (__SUBJ__ / __OBJ__)."""
    text = row["text"]
    e1, e1_t = row["subj"], row["subj_type"]
    e2, e2_t = row["obj"], row["obj_type"]

    masked = text
    spans = [s for s in {e1, e2} if isinstance(s, str) and s]
    for span in sorted(spans, key=len, reverse=True):
        if span in masked:
            placeholder = "__SUBJ__" if span == e1 else "__OBJ__"
            masked = masked.replace(span, placeholder)
    return f"{e1_t} REL {e2_t} | {masked}"


def build_feature_types_only(row):
    """3. ENTITY_TYPES_ONLY: Only entity type pair (e.g. 'APT REL MAL')."""
    return f"{row['subj_type']} REL {row['obj_type']}"


def build_feature_pure_context(row):
    """4. PURE_CONTEXT_ONLY: Masked sentence text ONLY (no entity types prefix)."""
    text = row["text"]
    e1 = row["subj"]
    e2 = row["obj"]

    masked = text
    spans = [s for s in {e1, e2} if isinstance(s, str) and s]
    for span in sorted(spans, key=len, reverse=True):
        if span in masked:
            placeholder = "__SUBJ__" if span == e1 else "__OBJ__"
            masked = masked.replace(span, placeholder)
    return masked


# ── Model Evaluator ────────────────────────────────────────────────────────────

def run_variant(name, feature_fn, train, val):
    X_train_text = [feature_fn(r) for r in train]
    y_train = [r["relation"] for r in train]
    X_val_text = [feature_fn(r) for r in val]
    y_val = [r["relation"] for r in val]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=60_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("norm", Normalizer()),
        ("clf", LogisticRegression(
            C=5.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )),
    ])
    pipeline.fit(X_train_text, y_train)

    preds = pipeline.predict(X_val_text)
    acc = accuracy_score(y_val, preds)

    majority_class = Counter(y_train).most_common(1)[0][0]
    majority_acc = accuracy_score(y_val, [majority_class] * len(y_val))

    print(f"\n[{name}]")
    print(f"  Sample feature   : {X_train_text[0][:100]!r}")
    print(f"  Val accuracy     : {acc:.4f} ({100*acc:.1f}%)")
    print(f"  Majority baseline: {majority_acc:.4f} ({100*majority_acc:.1f}%)")
    print(f"  Delta over maj   : {(acc - majority_acc)*100:+.1f} pp")

    return acc


def main():
    print("=" * 70)
    print("  Entity-Leakage & Feature Ablation Diagnostic (Fully Converged)")
    print("=" * 70)
    print(f"\nLoading dataset: {DATA_PATH}")
    data = load_data()

    train, val, test = text_dedup_split(data)
    # ── Regression Guard ─────────────────────────────────────────────────────
    # These counts MUST match train_relation_extractor.py's load_tire_data_dedup()
    # which trained fast_extractor.pkl. If this assertion fires, the split logic
    # has drifted from the canonical split. Check: seed, RNG API, text-set scope,
    # and rounding — all three bugs are documented in the git history (a28b711).
    _EXPECTED_TRAIN = 43_996
    _EXPECTED_VAL   =  8_519
    assert len(train) == _EXPECTED_TRAIN, (
        f"Split regression: train={len(train):,} != {_EXPECTED_TRAIN:,}. "
        f"Split logic must match train_relation_extractor.load_tire_data_dedup()"
    )
    assert len(val) == _EXPECTED_VAL, (
        f"Split regression: val={len(val):,} != {_EXPECTED_VAL:,}. "
        f"Split logic must match train_relation_extractor.load_tire_data_dedup()"
    )
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  Train instances: {len(train):,}  |  Val instances: {len(val):,}")

    print("\n" + "─" * 70)
    print("  Sample Feature Inspection (First 3 Val Instances)")
    print("─" * 70)
    for i in range(min(3, len(val))):
        row = val[i]
        print(f"\nInstance #{i+1} [True Relation: {row['relation']}]")
        print(f"  1. BASELINE          : {build_feature_baseline(row)!r}")
        print(f"  2. MASKED_TEXT       : {build_feature_masked(row)!r}")
        print(f"  3. ENTITY_TYPES_ONLY : {build_feature_types_only(row)!r}")
        print(f"  4. PURE_CONTEXT_ONLY : {build_feature_pure_context(row)!r}")

    print("\n" + "─" * 70)
    print("  Running 4-Way Feature Ablation Study...")
    print("─" * 70)

    acc_baseline     = run_variant("1. BASELINE (Types + Text + Context)", build_feature_baseline, train, val)
    acc_masked       = run_variant("2. MASKED_TEXT (Types + Masked Text)", build_feature_masked, train, val)
    acc_types_only   = run_variant("3. ENTITY_TYPES_ONLY (Types Pair Only)", build_feature_types_only, train, val)
    acc_pure_context = run_variant("4. PURE_CONTEXT_ONLY (Masked Context Only)", build_feature_pure_context, train, val)

    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY & LEAKAGE BREAKDOWN")
    print("=" * 70)
    print(f"  1. Baseline (Types + Text + Context) : {acc_baseline:.4f} ({100*acc_baseline:.1f}%)")
    print(f"  2. Masked Text (Types + Masked Context) : {acc_masked:.4f} ({100*acc_masked:.1f}%)")
    print(f"  3. Entity Types Only (No Context)     : {acc_types_only:.4f} ({100*acc_types_only:.1f}%)")
    print(f"  4. Pure Sentence Context (No Types)   : {acc_pure_context:.4f} ({100*acc_pure_context:.1f}%)")
    print("-" * 70)

    text_leakage_gap = acc_baseline - acc_masked
    type_signal = acc_types_only
    context_signal = acc_pure_context

    print(f"  Entity Text String Impact (1 - 2) : {text_leakage_gap*100:+.1f} pp")
    print(f"  Entity Type Signal (3 alone)      : {100*type_signal:.1f}% accuracy")
    print(f"  Pure Context Signal (4 alone)     : {100*context_signal:.1f}% accuracy")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())