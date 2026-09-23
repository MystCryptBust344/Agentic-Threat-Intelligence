"""
train_relation_extractor.py  —  Phase 1 v2 (corrected)
=======================================================
TF-IDF (1-3 ngrams) + Logistic Regression  —  trains in < 60 seconds.

Key fixes in this corrected version:
  1. TEXT-LEVEL DEDUP SPLIT (Fix 1): splits by UNIQUE SENTENCE, not by
     triplet instance. All triplets from a sentence go entirely to one
     split — no sentence appears in both train and val. This eliminates
     the 97.2% exact-duplicate leakage found in the instance-level split.
  2. HONEST ACCURACY REPORTING: val_acc reported relative to majority
     baseline computed on the SAME clean split. No pre-set target.
  3. NAMED THRESHOLDS: MIN_VIABLE_ACC = majority_baseline + 0.05 (hard
     floor that fails the build if model is barely better than guessing).
     Expected range 60-80% is documented in comments, not enforced.
  4. CONFUSION MATRIX saved as models/confusion_matrix.txt.
  5. PER-CLASS N printed post-split so zero-instance classes are visible.
  6. Collapse guard unchanged — still asserts std > 0.03 and unique > 20.

Usage:
    python -X utf8 train_relation_extractor.py
"""

import json
import os
import pickle
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# ── Config ─────────────────────────────────────────────────────────────────────
TIRE_JSON        = "../TIRE/dnrti_aug_stix2_je.json"
MODEL_DIR        = "models"
FAST_MODEL_PATH  = os.path.join(MODEL_DIR, "fast_extractor.pkl")
LABEL_MAP_FILE   = os.path.join(MODEL_DIR, "label_map.json")
CM_PATH          = os.path.join(MODEL_DIR, "confusion_matrix.txt")

# Hard collapse-guard thresholds (fail the build if not met)
MIN_UNIQUE_CONFS = 20     # distinct confidence values (rounded to 3dp) required
MIN_CONF_STD     = 0.03   # minimum confidence std-dev across validation set

# MIN_VIABLE_ACC is a build-failure floor, NOT a reporting target.
# Expected range after text-dedup split is 60-80% (informational).
# Set at runtime as: majority_baseline + MIN_VIABLE_DELTA
MIN_VIABLE_DELTA = 0.05   # model must beat majority by at least 5pp or build fails


# ── Feature Engineering ────────────────────────────────────────────────────────
def _build_feature_str(text: str, e1: str, e1_type: str,
                        e2: str, e2_type: str) -> str:
    """
    Produce a rich feature string for TF-IDF by:
      1. Inserting entity markers [E1]...[/E1] and [E2]...[/E2] into the sentence.
      2. Prepending the entity-type pair as a prefix token (e.g. "APT REL MAL |").
    Both the lexical context and the entity-type pair are important signals.
    """
    marked = text
    if e1 and e1 in marked:
        marked = marked.replace(e1, f"[E1] {e1} [/E1]", 1)
    if e2 and e2 in marked and e1 != e2:
        marked = marked.replace(e2, f"[E2] {e2} [/E2]", 1)
    return f"{e1_type} REL {e2_type} | {marked}"


# ── Text-Dedup Split (Fix 1) ───────────────────────────────────────────────────
def load_tire_data_dedup(tire_path: str,
                         val_frac: float = 0.15,
                         test_frac: float = 0.15,
                         seed: int = 42
                         ) -> Tuple[List, List, List, List, List, List, Dict]:
    """
    Text-deduplicated split: all triplets from the same sentence go entirely
    to one split (train, val, or test). No sentence appears in more than one
    split, eliminating the 97.2% instance-level leakage found in v2.

    Returns X_train, X_val, X_test, y_train, y_val, y_test, split_info.
    """
    with open(tire_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Group all instances by unique text
    text_to_instances: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for item in raw:
        text = item.get("text", "")
        ents = item.get("entities", [])
        for rel in item.get("relations", []):
            rt, i1, i2 = rel[0], rel[1], rel[2]
            if i1 >= len(ents) or i2 >= len(ents):
                continue
            e1 = ents[i1]; e2 = ents[i2]
            feat = _build_feature_str(
                text,
                e1[2] if len(e1) > 2 else "",
                e1[3] if len(e1) > 3 else "UNK",
                e2[2] if len(e2) > 2 else "",
                e2[3] if len(e2) > 3 else "UNK",
            )
            text_to_instances[text].append((feat, rt))

    unique_texts = list(text_to_instances.keys())
    n_texts = len(unique_texts)
    print(f"  Unique sentences in TIRE     : {n_texts:,}")
    print(f"  Total triplet instances      : {sum(len(v) for v in text_to_instances.values()):,}")
    print(f"  (Exact-duplicate texts skipped at split time — all go to one split)")

    # Deterministic shuffle of unique texts
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_texts).tolist()

    n_val  = int(n_texts * val_frac)
    n_test = int(n_texts * test_frac)
    n_train = n_texts - n_val - n_test

    texts_train = shuffled[:n_train]
    texts_val   = shuffled[n_train:n_train + n_val]
    texts_test  = shuffled[n_train + n_val:]

    def expand(texts):
        X, y = [], []
        for t in texts:
            for feat, label in text_to_instances[t]:
                X.append(feat); y.append(label)
        return X, y

    X_train, y_train = expand(texts_train)
    X_val,   y_val   = expand(texts_val)
    X_test,  y_test  = expand(texts_test)

    split_info = {
        "n_unique_texts": n_texts,
        "n_train_texts": len(texts_train),
        "n_val_texts":   len(texts_val),
        "n_test_texts":  len(texts_test),
        "texts_train_set": set(texts_train),
        "texts_val_set":   set(texts_val),
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, split_info


# ── Collapse Guard ─────────────────────────────────────────────────────────────
def _collapse_guard(pipeline: Pipeline, X_val: List[str]) -> Dict:
    """
    Hard assertion that fires BEFORE the model is saved.
    If either check fails, AssertionError is raised — nothing is written to disk.

    Rationale: a model can appear to train successfully yet produce near-constant
    confidence scores (e.g. always 0.87) due to class imbalance + bad LR geometry.
    The diagnostic printout in diagnostics.py catches this after the fact; this
    guard prevents the broken model from ever being saved in the first place.
    """
    proba     = pipeline.predict_proba(X_val)
    max_confs = proba.max(axis=1)
    unique    = len(set(max_confs.round(3)))
    std       = float(max_confs.std())
    mean_conf = float(max_confs.mean())

    sep = "-" * 50
    print(f"\n  {sep}")
    print(f"  [Collapse Guard]")
    print(f"    Unique confidence values (3dp): {unique:5d}   threshold > {MIN_UNIQUE_CONFS}")
    print(f"    Confidence std-dev:             {std:.4f}   threshold > {MIN_CONF_STD}")
    print(f"    Confidence mean:                {mean_conf:.4f}")

    assert unique > MIN_UNIQUE_CONFS, (
        f"\n\n{'!'*60}\n"
        f"  COLLAPSE DETECTED — MODEL NOT SAVED\n"
        f"  Only {unique} unique confidence values (need > {MIN_UNIQUE_CONFS}).\n"
        f"  The model outputs near-constant probabilities across inputs.\n"
        f"  Suggested fixes: increase C, reduce min_df, check class balance.\n"
        f"{'!'*60}"
    )
    assert std > MIN_CONF_STD, (
        f"\n\n{'!'*60}\n"
        f"  COLLAPSE DETECTED — MODEL NOT SAVED\n"
        f"  Confidence std = {std:.4f} (need > {MIN_CONF_STD}).\n"
        f"  The model has negligible confidence variance across inputs.\n"
        f"  Suggested fixes: different solver, increase max_iter, review features.\n"
        f"{'!'*60}"
    )

    print(f"    Guard PASSED  — model is discriminating (not collapsed)")
    print(f"  {sep}")
    return {"unique": unique, "std": std, "mean": mean_conf,
            "min": float(max_confs.min()), "max": float(max_confs.max())}


# ── Training ───────────────────────────────────────────────────────────────────
def train_fast_model() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Phase 1 v2 (corrected)  —  Relation Extractor  (TF-IDF + LR)")
    print("=" * 60)

    # ── 1. Load + Text-Dedup Split ─────────────────────────────────────────
    print(f"\n[1/6] Loading TIRE dataset + text-deduplicated split:  {TIRE_JSON}")
    X_train, X_val, X_test, y_train, y_val, y_test, split_info = \
        load_tire_data_dedup(TIRE_JSON)

    print(f"\n  Text-dedup split summary (no sentence in >1 split):")
    print(f"    Unique texts → Train : {split_info['n_train_texts']:,}"
          f"  → {len(X_train):,} instances")
    print(f"    Unique texts → Val   : {split_info['n_val_texts']:,}"
          f"  → {len(X_val):,} instances")
    print(f"    Unique texts → Test  : {split_info['n_test_texts']:,}"
          f"  → {len(X_test):,} instances")

    # Verify zero overlap
    train_text_set = split_info["texts_train_set"]
    val_text_set   = split_info["texts_val_set"]
    overlap = len(train_text_set & val_text_set)
    assert overlap == 0, f"SPLIT OVERLAP BUG: {overlap} texts in both train and val!"
    print(f"    Train/val sentence overlap : 0 (verified \u2713)")

    # ── 2. Label map ─────────────────────────────────────────────────────────
    all_labels = sorted(set(y_train + y_val + y_test))
    label_map  = {lbl: i for i, lbl in enumerate(all_labels)}
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n  {len(label_map)} classes saved → {LABEL_MAP_FILE}")

    # ── 3. Per-class N post-split ──────────────────────────────────────────
    print(f"\n[2/6] Per-class instance count after text-dedup split:")
    val_counts  = Counter(y_val)
    train_counts = Counter(y_train)
    print(f"  {'Class':<25} {'Train':>7} {'Val':>7}  Note")
    print(f"  {'-'*25} {'-'*7} {'-'*7}")
    for lbl in all_labels:
        n_tr = train_counts.get(lbl, 0)
        n_vl = val_counts.get(lbl, 0)
        note = ""
        if n_vl == 0:
            note = "  *** ZERO VAL INSTANCES — class invisible to evaluation ***"
        elif n_vl < 15:
            note = f"  [LOW-N: only {n_vl} val instances — treat with caution]"
        print(f"  {lbl:<25} {n_tr:>7,} {n_vl:>7,}{note}")

    # ── 4. Majority-class baseline ─────────────────────────────────────────
    print(f"\n[3/6] Majority-class baseline on validation set (same clean split)...")
    majority_label = val_counts.most_common(1)[0][0]
    majority_acc   = val_counts[majority_label] / len(y_val)
    random_acc     = 1.0 / len(all_labels)

    # MIN_VIABLE_ACC: hard build-failure floor (not a reporting target)
    MIN_VIABLE_ACC = majority_acc + MIN_VIABLE_DELTA
    print(f"  Majority class    : '{majority_label}'  {val_counts[majority_label]:,}/{len(y_val):,}"
          f"  =  {100*majority_acc:.1f}%")
    print(f"  Random baseline   : {100*random_acc:.1f}%")
    print(f"  MIN_VIABLE_ACC    : {100*MIN_VIABLE_ACC:.1f}%  (majority + {100*MIN_VIABLE_DELTA:.0f}pp"
          f"  — fails build if not met)")
    print(f"  Expected range    : 60-80%  (informational, not enforced)")

    # ── 5. Train ─────────────────────────────────────────────────────────
    print(f"\n[4/6] Training TF-IDF + LogisticRegression...")
    t0 = time.time()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=60_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f}s")

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    print(f"\n[5/6] Evaluating on validation set...")
    y_pred    = pipeline.predict(X_val)
    model_acc = float((np.array(y_pred) == np.array(y_val)).mean())
    delta     = model_acc - majority_acc

    print(f"\n  Model accuracy    : {model_acc:.4f}  ({100*model_acc:.1f}%)")
    print(f"  Majority baseline : {majority_acc:.4f}  ({100*majority_acc:.1f}%)  [same clean split]")
    print(f"  Random baseline   : {random_acc:.4f}  ({100*random_acc:.1f}%)")

    if delta > 0:
        print(f"  => Model BEATS majority baseline by  +{100*delta:.1f} pp")
    else:
        print(f"  => *** Model WORSE than majority baseline by {-100*delta:.1f} pp ***")
        print(f"     TF-IDF bag-of-words cannot separate these classes on the honest split.")

    # Hard floor check (separate from expected range)
    if model_acc < MIN_VIABLE_ACC:
        raise AssertionError(
            f"\n{'!'*60}\n"
            f"  BUILD FAIL: val_acc={model_acc:.4f} < MIN_VIABLE_ACC={MIN_VIABLE_ACC:.4f}\n"
            f"  Model does not meet the minimum viable bar (majority + {100*MIN_VIABLE_DELTA:.0f}pp).\n"
            f"{'!'*60}"
        )

    print(f"\n  Per-class F1 report:")
    print(classification_report(y_val, y_pred, target_names=all_labels, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=all_labels)
    header = ",".join(all_labels)
    np.savetxt(CM_PATH, cm, fmt="%4d", delimiter=",",
               header=header, comments="")
    print(f"  Confusion matrix saved → {CM_PATH}")

    # Flag classes with < 15 val instances in the confusion matrix
    for lbl, n_val_instances in zip(all_labels, cm.sum(axis=1)):
        if 0 < n_val_instances < 15:
            print(f"  [LOW-N WARNING] '{lbl}': {n_val_instances} val instances"
                  f" — treat per-class F1 with caution")
        elif n_val_instances == 0:
            print(f"  [ZERO-N WARNING] '{lbl}': 0 val instances — class absent from evaluation")

    # ── 7. Collapse guard (hard assertion) ────────────────────────────────
    print(f"\n[6/6] Collapse-detection guard (model NOT saved if this fails)...")
    guard = _collapse_guard(pipeline, X_val)

    # Confidence histogram (on raw model output — not post-weight)
    proba     = pipeline.predict_proba(X_val)
    max_confs = proba.max(axis=1)
    buckets   = [0.0, 0.50, 0.70, 0.90, 1.01]
    labels_h  = ["< 0.50 (REJECTED)", "0.50-0.70 (FLAGGED)",
                  "0.70-0.90 (ACCEPT-partial)", ">= 0.90 (ACCEPT-full)"]
    print(f"\n  Confidence histogram on {len(max_confs):,} val instances"
          f" (raw model output — policy thresholds shown):")
    for i, (lo, hi) in enumerate(zip(buckets, buckets[1:])):
        mask  = (max_confs >= lo) & (max_confs < hi)
        count = int(mask.sum())
        pct   = 100 * count / len(max_confs)
        bar   = "█" * min(35, int(35 * count / max(len(max_confs), 1)))
        print(f"  [{labels_h[i]:30s}]  {bar:<35s}  {pct:5.1f}%  ({count:,})")

    # ── 8. Save ────────────────────────────────────────────────────────────
    payload = {
        "pipeline"      : pipeline,
        "label_map"     : label_map,
        "classes"       : all_labels,
        "val_acc"       : model_acc,
        "majority_acc"  : majority_acc,
        "min_viable_acc": majority_acc + MIN_VIABLE_DELTA,
        "guard"         : guard,
        "split_info"    : {k: v for k, v in split_info.items()
                           if not k.endswith("_set")},  # sets not picklable
    }
    with open(FAST_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n  Model saved  ->  {FAST_MODEL_PATH}")
    print(f"\n{'=' * 60}")
    print(f"  DONE.  val_acc={model_acc:.4f}  "
          f"({'BEATS' if delta > 0 else 'LOSES TO'} majority baseline "
          f"by {abs(100*delta):.1f} pp on honest text-dedup split)")
    print(f"  Expected range: 60-80% (informational)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    train_fast_model()
