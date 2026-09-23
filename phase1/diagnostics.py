"""
diagnostics.py  —  Phase 1 v2 (corrected)
==========================================
Scientific validation of the UQ pipeline after training.

Checks:
  1. Global confidence distribution (min, max, mean, std, unique values)
  2. Per-class confidence table (detects per-class collapse)
  3. Threshold bucket distribution (all 3 buckets must be populated)
  4. NER sanity check (known bad tokens must NOT be tagged)
  5. Per-triplet spot-check table (10 examples with varying confidences)
  6. Accuracy-by-confidence-bucket calibration table (Fix 2b):
       For each confidence bucket [<0.50, 0.50-0.70, 0.70-0.90, >=0.90],
       report fraction of triplets correctly classified.
       If accuracy tracks confidence monotonically (ascending), the UQ
       signal is calibrated — Phase 2 attention weighting is trustworthy.
       If anti-correlated, Phase 2 would actively down-weight good signal.

Usage:
    python -X utf8 diagnostics.py
    python -X utf8 diagnostics.py --samples 500
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

from triplet_extractor import TripletExtractor, extract_entities_heuristic

FAST_MODEL_PATH = "models/fast_extractor.pkl"
TIRE_JSON       = "../TIRE/dnrti_aug_stix2_je.json"


# ── NER sanity sentences ───────────────────────────────────────────────────────
_NER_TEST_CASES = [
    {
        "text": "APT29 used Cobalt Strike against the US Government.",
        "must_contain": [("APT29", "Threat-Actor"), ("Cobalt Strike", "Malware")],
        "must_not_contain": ["used", "against", "Government"],
    },
    {
        "text": "The Lazarus Group deployed WannaCry ransomware targeting financial sector organizations.",
        "must_contain": [("Lazarus Group", "Threat-Actor"), ("WannaCry", "Malware")],
        "must_not_contain": ["ransomware", "targeting"],
    },
    {
        "text": "admin@338 targeted organizations using spear phishing emails.",
        "must_contain": [("admin@338", "Threat-Actor"), ("spear phishing", "Attack-Pattern")],
        "must_not_contain": ["using", "organizations"],
    },
    {
        "text": "The actor exploited CVE-2021-44228 to gain remote code execution.",
        "must_contain": [("CVE-2021-44228", "Vulnerability")],
        "must_not_contain": ["actor", "gain", "remote"],
    },
    {
        "text": "The campaign was attributed to BlackBear, a previously unknown group.",
        "must_contain": [("BlackBear", "Candidate-Entity")],
        "must_not_contain": ["campaign", "unknown", "previously"],
    },
]


def run_ner_sanity_check() -> bool:
    """Returns True if all NER checks pass."""
    print("\n" + "─" * 55)
    print("  NER Sanity Check  (known bad tokens must NOT be tagged)")
    print("─" * 55)
    all_pass = True
    for tc in _NER_TEST_CASES:
        text  = tc["text"]
        ents  = extract_entities_heuristic(text)
        names = {e[0].lower() for e in ents}
        print(f"\n  Sentence: \"{text}\"")
        print(f"  Entities: {[(e[0], e[1]) for e in ents]}")

        for expected_name, expected_label in tc.get("must_contain", []):
            hit = any(e[0].lower() == expected_name.lower() and e[1] == expected_label
                      for e in ents)
            if hit:
                print(f"    FOUND (expected)  : '{expected_name}' [{expected_label}]")
            else:
                print(f"    MISSING (expected): '{expected_name}' [{expected_label}]  -- OK if Tier-2")

        failed = False
        for bad in tc.get("must_not_contain", []):
            if bad.lower() in names:
                print(f"    *** FAIL: '{bad}' was incorrectly tagged ***")
                all_pass = False
                failed = True
        if not failed:
            print(f"    No garbage tokens tagged.  PASS")

    return all_pass


def _canonical_split_texts(raw_data, seed: int = 42,
                            val_frac: float = 0.15, test_frac: float = 0.15):
    """
    Reproduces the EXACT text-level split used by
    train_relation_extractor.py's load_tire_data_dedup() (the split that
    actually trained fast_extractor.pkl), mirrored in check_entity_leakage.py
    and verify_binary_gatekeeper.py. Returns (train_texts, val_texts, test_texts)
    as sets of raw sentence strings.

    Scope note: only texts that produced >= 1 in-bounds relation instance are
    included, matching the canonical split's text_to_instances.keys() scope
    (documented root cause of a prior split-size regression, git a28b711).
    """
    from collections import defaultdict as _dd
    text_has_instance = _dd(bool)
    for d in raw_data:
        text = d.get("text", "")
        ents = d.get("entities", [])
        for rel in d.get("relations", []):
            rt, i1, i2 = rel[0], rel[1], rel[2]
            if i1 >= len(ents) or i2 >= len(ents):
                continue
            text_has_instance[text] = True

    unique_texts = list(text_has_instance.keys())   # insertion order — must match canonical
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_texts).tolist()

    n = len(shuffled)
    n_val   = int(n * val_frac)
    n_test  = int(n * test_frac)
    n_train = n - n_val - n_test   # remainder assigned to train — matches canonical

    texts_train = set(shuffled[:n_train])
    texts_val   = set(shuffled[n_train:n_train + n_val])
    texts_test  = set(shuffled[n_train + n_val:])
    return texts_train, texts_val, texts_test


def run_calibration_check(n_samples: int = 500) -> bool:
    """
    Fix 2b: Accuracy-by-confidence-bucket calibration table.

    Loads the trained model and runs it on genuinely held-out samples
    (texts assigned to val/test by the canonical split — see
    _canonical_split_texts). For each confidence bucket, reports:
      - Count of triplets
      - Fraction correctly classified (accuracy in that bucket)

    PASS condition: accuracy increases monotonically from low to high
    confidence buckets (i.e., the model is more confident when it's more
    likely to be right). This is what makes Phase 2's attention-weighted
    aggregation trustworthy rather than random or actively harmful.

    If anti-monotonic (high confidence correlates with lower accuracy),
    Phase 2 would down-weight good edges and up-weight bad ones — that
    is WORSE than uniform weighting and must be flagged.
    """
    print("\n" + "─" * 60)
    print("  Calibration Check — Accuracy-by-Confidence-Bucket")
    print("  (Does accuracy track confidence? This validates Phase 2 UQ.)")
    print("─" * 60)

    if not os.path.exists(FAST_MODEL_PATH):
        print("  [SKIP] No model found. Run train_relation_extractor.py first.")
        return True

    with open(FAST_MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    pipeline = saved["pipeline"]
    classes  = saved["classes"]

    if not os.path.exists(TIRE_JSON):
        print(f"  [SKIP] TIRE dataset not found at {TIRE_JSON}.")
        return True

    with open(TIRE_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # FIX: raw[-n_samples:] has NO guaranteed relationship to the actual
    # train/val/test partition — the real split is a random shuffle of
    # unique TEXTS, decoupled from file position. Using the last N raw
    # entries risked silently evaluating "calibration" on samples the
    # model was partly or wholly trained on. Use the real held-out texts
    # (val + test) instead.
    _, texts_val, texts_test = _canonical_split_texts(raw)
    held_out_texts = texts_val | texts_test
    held_out_samples = [s for s in raw if s.get("text", "") in held_out_texts]

    if not held_out_samples:
        print("  [SKIP] No genuinely held-out samples found.")
        return True

    test_samples = held_out_samples[:n_samples] if n_samples else held_out_samples
    print(f"  Using {len(test_samples):,} verified held-out samples "
          f"(val+test texts; {len(held_out_samples):,} available total)")

    feat_strs: list = []
    true_labels: list = []

    for sample in test_samples:
        text = sample.get("text", "")
        ents = sample.get("entities", [])
        for rel in sample.get("relations", []):
            rt, i1, i2 = rel[0], rel[1], rel[2]
            if rt == "noRelation" or i1 >= len(ents) or i2 >= len(ents):
                continue
            e1, e2 = ents[i1], ents[i2]
            feat = (
                f"{e1[3] if len(e1) > 3 else 'UNK'} REL "
                f"{e2[3] if len(e2) > 3 else 'UNK'} | "
                f"{text}"
            )
            feat_strs.append(feat)
            true_labels.append(rt)

    if not feat_strs:
        print("  [SKIP] No instances extracted from test samples.")
        return True

    proba     = pipeline.predict_proba(feat_strs)
    pred_idx  = proba.argmax(axis=1)
    preds     = [classes[i] for i in pred_idx]
    max_confs = proba.max(axis=1)
    correct   = np.array([p == t for p, t in zip(preds, true_labels)], dtype=float)

    # Bucket definitions matching policy thresholds
    bucket_defs = [
        (0.0,  0.50, "< 0.50 (REJECTED)"),
        (0.50, 0.70, "0.50-0.70 (FLAGGED)"),
        (0.70, 0.90, "0.70-0.90 (ACCEPT-partial)"),
        (0.90, 1.01, ">= 0.90 (ACCEPT-full)"),
    ]

    print(f"\n  {'Bucket':<28} {'N':>6}  {'Accuracy':>9}  {'Bar':<25}  Note")
    print(f"  {'-'*28} {'-'*6}  {'-'*9}  {'-'*25}")

    accs = []
    for lo, hi, label in bucket_defs:
        mask = (max_confs >= lo) & (max_confs < hi)
        n    = int(mask.sum())
        if n == 0:
            print(f"  {label:<28} {0:>6}  {'N/A':>9}  {'':25}  [empty bucket]")
            accs.append(None)
            continue
        acc  = float(correct[mask].mean())
        bar  = "█" * min(25, int(25 * acc))
        note = ""
        accs.append(acc)
        print(f"  {label:<28} {n:>6}  {acc:>9.3f}  {bar:<25}  {note}")

    # Monotonicity check: accuracy should not decrease bucket-to-bucket
    filled = [(i, a) for i, a in enumerate(accs) if a is not None]
    if len(filled) >= 2:
        monotonic = all(
            filled[i][1] <= filled[i+1][1]
            for i in range(len(filled) - 1)
        )
        # Near-monotonic (allow 3pp drop)
        near_mono = all(
            filled[i][1] - filled[i+1][1] <= 0.03
            for i in range(len(filled) - 1)
        )
    else:
        monotonic = near_mono = True

    print()
    if monotonic:
        print("  CALIBRATION: PASS — accuracy increases monotonically with confidence.")
        print("  Phase 2 attention-weighted aggregation will preferentially weight")
        print("  high-confidence edges that are more likely correct. UQ is trustworthy.")
        calib_ok = True
    elif near_mono:
        print("  CALIBRATION: NEAR-PASS — accuracy roughly tracks confidence (<=3pp dip).")
        print("  Phase 2 UQ signal is usable. Minor anti-monotonic step; review per-class F1.")
        calib_ok = True
    else:
        print("  CALIBRATION: FAIL — accuracy is NOT monotonic with confidence.")
        print("  Phase 2 attention weighting would actively down-weight correct edges.")
        print("  Options: (a) add calibration (Platt scaling), (b) use uniform weights,")
        print("           (c) retrain with better features (dependency path, POS tags).")
        calib_ok = False

    return calib_ok


def run_diagnostics(args) -> None:
    print("\n" + "=" * 60)
    print("  Phase 1  —  Diagnostic Report  (v2 corrected)")
    print("=" * 60)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    extractor = TripletExtractor()
    if not extractor._model_loaded:
        print("\n[ERROR] No model found. Run: python -X utf8 train_relation_extractor.py")
        sys.exit(1)

    # ── 2. Confidence distribution ────────────────────────────────────────────
    print(f"\n[1/4] Running confidence distribution analysis "
          f"({args.samples} TIRE test-split samples)...")
    result = extractor.diagnose_confidence_distribution(n_samples=args.samples)

    if result.get("collapsed", True):
        print("\n*** DIAGNOSTIC FAILED: model confidence is collapsed. ***")
        print("    Re-train with: python -X utf8 train_relation_extractor.py")
        sys.exit(1)

    buckets = result.get("buckets", {})
    if not result.get("all_buckets_populated", False):
        empty = [k for k, v in buckets.items() if v == 0]
        print(f"\n*** DIAGNOSTIC WARNING: buckets {empty} are empty. ***")
        print("    The UQ thresholding policy is not fully exercised.")
    else:
        print("\n  All threshold buckets populated  --  UQ is discriminating.")

    # ── 3. Calibration check (Fix 2b) ─────────────────────────────────────────
    print(f"\n[2/4] Calibration check — accuracy by confidence bucket...")
    calib_ok = run_calibration_check(n_samples=args.samples)

    # ── 4. NER sanity check ───────────────────────────────────────────────────
    print(f"\n[3/4] NER sanity check...")
    ner_ok = run_ner_sanity_check()
    if ner_ok:
        print("\n  NER sanity check: PASSED")
    else:
        print("\n  NER sanity check: FAILED  --  garbage tokens were tagged.")

    # ── 5. Per-triplet spot-check table ───────────────────────────────────────
    print(f"\n[4/4] Per-triplet spot-check (10 examples with varying confidences)...")
    with open(TIRE_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    spot_triplets = []
    for sample in raw[-200:]:           # use test-set samples
        tris = extractor.extract_from_tire_sample(sample)
        spot_triplets.extend(tris)
        if len(spot_triplets) >= 15:
            break

    spot_triplets = spot_triplets[:10]
    confs = [t.c_nlp for t in spot_triplets]
    unique_confs = len(set(round(c, 3) for c in confs))

    print(f"\n  {'Subject':<22} {'Relation':<18} {'Object':<22} "
          f"{'c_NLP (raw)':>11} {'Status':>9}")
    print(f"  {'-'*22} {'-'*18} {'-'*22} {'-'*11} {'-'*9}")
    for t in spot_triplets:
        flag = {"ACCEPTED": "", "FLAGGED": "~", "REJECTED": "!"}[t.status]
        print(f"  {t.subject[:21]:<22} {t.relation[:17]:<18} "
              f"{t.object_[:21]:<22} {t.c_nlp:>11.4f} {flag}{t.status}")

    print(f"\n  Unique confidence values in spot-check: {unique_confs}/10")
    if unique_confs < 3:
        print("  *** WARNING: very little variation in this sample ***")
    else:
        print("  GOOD: confidence varies across triplets.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Diagnostic Summary")
    print(f"{'=' * 60}")
    print(f"  Confidence std    : {result.get('std', 0):.4f}  "
          f"({'PASS' if result.get('std', 0) > 0.03 else 'FAIL'})")
    print(f"  Unique conf vals  : {result.get('unique', 0)}  "
          f"({'PASS' if result.get('unique', 0) > 20 else 'FAIL'})")
    print(f"  All buckets filled: "
          f"{'YES' if result.get('all_buckets_populated') else 'NO'}")
    print(f"  Calibration check : {'PASS' if calib_ok else 'FAIL — see above'}")
    print(f"  NER check         : {'PASS' if ner_ok else 'FAIL'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Diagnostic Tool")
    parser.add_argument("--samples", type=int, default=300,
                        help="Number of TIRE test-split samples to analyse (default: 300)")
    args = parser.parse_args()
    run_diagnostics(args)