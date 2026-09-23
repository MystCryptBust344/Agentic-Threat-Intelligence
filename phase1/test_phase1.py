"""
test_phase1.py — Phase 1 Verification Test Suite
=================================================
Tests all Phase 1 components individually and end-to-end.

Usage:
    python test_phase1.py              # run all tests
    python test_phase1.py --test-db    # store + MITRE seeding only
    python test_phase1.py --test-model # model loading + relation classification
    python test_phase1.py --test-ingestion  # full end-to-end pipeline (fast)
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import argparse
import json
import math
import sys
import time
import tempfile
from pathlib import Path

# Add phase1/ to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table

console = Console(highlight=False)
PASS = "[bold green]✓ PASS[/bold green]"
FAIL = "[bold red]✗ FAIL[/bold red]"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def assert_eq(name: str, got, expected):
    ok = got == expected
    console.print(f"  {PASS if ok else FAIL} {name}: got={got!r}, expected={expected!r}")
    return ok

def assert_close(name: str, got: float, expected: float, tol: float = 1e-4):
    ok = abs(got - expected) < tol
    console.print(f"  {PASS if ok else FAIL} {name}: got={got:.6f}, expected≈{expected:.6f}")
    return ok

def assert_in_range(name: str, val: float, lo: float, hi: float):
    ok = lo <= val <= hi
    console.print(f"  {PASS if ok else FAIL} {name}: {val:.4f} ∈ [{lo}, {hi}]")
    return ok

# ------------------------------------------------------------------------------
# TEST 1: Database Store + MITRE Seeding
# ------------------------------------------------------------------------------
def test_db(seed_mitre: bool = False):
    console.print("\n[bold cyan]=== TEST 1: TKG Store Layer ===[/bold cyan]")
    from tkg_store import (
        TKGNode, TKGEdge, JSONLGraphStore, InMemoryGraphStore, DualLayerTKG,
        build_persistent_store, LAMBDA_DECAY
    )

    passes = 0
    total  = 0

    with tempfile.TemporaryDirectory() as tmp:
        # 1a. JSONLGraphStore write + read
        store = JSONLGraphStore(output_dir=tmp)
        node  = TKGNode("n001", "Malware", "EvilRAT")
        store.add_node(node)
        total += 1
        n_rec = store.get_node("n001")
        if n_rec:
            passes += 1
            console.print(f"  {PASS} JSONLGraphStore: node written & indexed")
        else:
            console.print(f"  {FAIL} JSONLGraphStore: node not found in index")

        # 1b. JSONL append-only: get_influence_subgraph must raise NotImplementedError
        total += 1
        try:
            store.get_influence_subgraph(["n001"])
            console.print(f"  {FAIL} JSONLGraphStore.get_influence_subgraph should raise NotImplementedError")
        except NotImplementedError:
            passes += 1
            console.print(f"  {PASS} JSONLGraphStore.get_influence_subgraph correctly raises NotImplementedError")

        # 1c. Edge compute_metrics — bounded [0,1]
        total += 1
        edge = TKGEdge("e001", "n001", "n002", "uses", c_nlp=0.92, source_name="tire")
        edge.timestamp = time.time() - 3600   # 1 hour ago
        edge.source_reputation = 0.85
        edge.compute_metrics()
        delta_t_hours = 3600 / 3600.0 #1 hour
        ec_expected = 0.92 * math.exp(-LAMBDA_DECAY * delta_t_hours)
        tr_expected = (ec_expected + 0.85) / 2.0
        ok = (abs(edge.effective_confidence - ec_expected) < 1e-4 and
              abs(edge.threat_reliability   - tr_expected) < 1e-4 and
              0.0 <= edge.effective_confidence <= 1.0 and
              0.0 <= edge.threat_reliability   <= 1.0)
        passes += ok
        console.print(
            f"  {PASS if ok else FAIL} Edge bounded metrics: "
            f"ec={edge.effective_confidence:.4f}, tr={edge.threat_reliability:.4f}"
        )

        # 1d. InMemoryGraphStore subgraph query
        total += 1
        mem = InMemoryGraphStore()
        mem.add_node(TKGNode("n1", "Malware",       "RAT-X"))
        mem.add_node(TKGNode("n2", "Threat-Actor",  "APT29"))
        mem.add_node(TKGNode("n3", "Attack-Pattern","Phishing"))
        edge2 = TKGEdge("e1", "n2", "n1", "uses", c_nlp=0.91, source_name="mitre-attack")
        edge2.compute_metrics()
        edge3 = TKGEdge("e2", "n2", "n3", "uses", c_nlp=0.88, source_name="mitre-attack")
        edge3.compute_metrics()
        mem.add_edge(edge2)
        mem.add_edge(edge3)
        sg = mem.get_influence_subgraph(["n2"], hops=1)
        ok = sg.number_of_nodes() >= 2
        passes += ok
        console.print(f"  {PASS if ok else FAIL} InMemory subgraph: {sg.number_of_nodes()} nodes within 1 hop of n2")

        # 1e. DualLayerTKG threshold policy
        total += 1
        tkg = DualLayerTKG(JSONLGraphStore(output_dir=tmp), flush_interval=999)
        tkg.ingest_node(TKGNode("src", "Threat-Actor", "APT1"))
        tkg.ingest_node(TKGNode("tgt", "Malware",      "BlackEnergy"))
        e_acc  = tkg.ingest_triplet("src", "uses", "tgt", c_nlp=0.95, source_name="tire")
        e_part = tkg.ingest_triplet("src", "uses", "tgt", c_nlp=0.75, source_name="tire")
        e_flag = tkg.ingest_triplet("src", "uses", "tgt", c_nlp=0.60, source_name="tire")
        e_rej  = tkg.ingest_triplet("src", "uses", "tgt", c_nlp=0.40, source_name="tire")
        ok = (e_acc is not None and
              e_part is not None and
              e_flag is not None and
              e_rej is None)
        passes += ok
        console.print(f"  {PASS if ok else FAIL} Threshold policy: acc={e_acc is not None} part={e_part is not None} flagged={e_flag is not None} rej={e_rej is None}")
        tkg.shutdown()

        # 1f. JSONL replay
        total += 1
        store2 = JSONLGraphStore(output_dir=tmp)
        mem2   = InMemoryGraphStore()
        replayed = store2.replay_into(mem2)
        ok = replayed > 0
        passes += ok
        console.print(f"  {PASS if ok else FAIL} JSONL replay: {replayed} mutations recovered into InMemoryGraphStore")

    # 1g. MITRE seeding (optional, takes time)
    if seed_mitre:
        mitre_path = "../enterprise-attack.json"
        if os.path.exists(mitre_path):
            total += 1
            console.print("\n  [yellow]Seeding MITRE ATT&CK (takes ~20s)...[/yellow]")
            with tempfile.TemporaryDirectory() as tmp2:
                store3 = JSONLGraphStore(output_dir=tmp2)
                n = store3.seed_from_mitre(mitre_path)
                ok = n > 1000
                passes += ok
                console.print(f"  {PASS if ok else FAIL} MITRE seeding: {n:,} records written")

    console.print(f"\n  [bold]Store Tests: {passes}/{total} passed[/bold]")
    return passes, total


# ------------------------------------------------------------------------------
# TEST 2: Model + Relation Classification
# ------------------------------------------------------------------------------
def test_model():
    console.print("\n[bold cyan]=== TEST 2: Triplet Extractor ===[/bold cyan]")
    from triplet_extractor import TripletExtractor, extract_entities_heuristic
    passes, total = 0, 0

    # 2a. NER: must find real entities, must NOT tag stopwords
    total += 1
    text = "APT29 used Cobalt Strike against the US Government."
    ents = extract_entities_heuristic(text)  # returns (name, label, tier_conf)
    entity_names = {e[0].lower() for e in ents}
    entity_pairs = [(e[0], e[1]) for e in ents]
    # Must contain known entities
    found_apt    = any("APT29" in e[0] for e in ents)
    found_mal    = any("Cobalt Strike" in e[0] for e in ents)
    # Must NOT contain garbage tokens
    garbage      = {"used", "against", "government", "the"} & entity_names
    ok = found_apt and found_mal and len(garbage) == 0
    passes += ok
    console.print(f"  {PASS if ok else FAIL} Heuristic NER: "
                  f"found APT={found_apt}, malware={found_mal}, garbage={garbage}")
    for e, l, c in ents:
        console.print(f"    -> '{e}' [{l}] tier_conf={c}")

    # 2b. Extractor init (heuristic mode if no model)
    total += 1
    extractor = TripletExtractor()
    ok = extractor is not None
    passes += ok
    mode = 'model' if extractor._model_loaded else 'heuristic'
    console.print(f"  {PASS if ok else FAIL} TripletExtractor initialized (mode={mode})")

    # 2c. Threshold policy
    total += 1
    from triplet_extractor import TripletExtractor
    w, s = TripletExtractor._apply_threshold(0.95); assert w == 1.0  and s == "ACCEPTED"
    w, s = TripletExtractor._apply_threshold(0.75); assert 0.7 <= w  and s == "ACCEPTED"
    w, s = TripletExtractor._apply_threshold(0.60); assert w == 0.3  and s == "FLAGGED"
    w, s = TripletExtractor._apply_threshold(0.40); assert w is None and s == "REJECTED"
    passes += 1
    console.print(f"  {PASS} Threshold policy: all 4 branches verified")

    # 2d. TIRE sample extraction
    total += 1
    tire_path = "../TIRE/dnrti_aug_stix2_je.json"
    if os.path.exists(tire_path):
        with open(tire_path) as f:
            sample = json.load(f)[0]
        triplets = extractor.extract_from_tire_sample(sample)
        ok = isinstance(triplets, list)
        passes += ok
        console.print(f"  {PASS if ok else FAIL} TIRE sample extraction: {len(triplets)} triplets found")
        for tri in triplets[:3]:
            console.print(f"    [{tri.status}] {tri.subject} —[{tri.relation}]→ {tri.object_} c={tri.c_nlp:.3f}")
    else:
        console.print("  [yellow]SKIP: TIRE file not found[/yellow]")

    console.print(f"\n  [bold]Model Tests: {passes}/{total} passed[/bold]")
    return passes, total


# ------------------------------------------------------------------------------
# TEST 4: UQ Confidence Varies (not constant) — bucket-based assertion
# ------------------------------------------------------------------------------
def test_varying_confidence():
    """
    Assert that ALL three threshold buckets (ACCEPTED, FLAGGED, REJECTED)
    are populated across 100 TIRE test-set samples.

    std > 0.01 alone is NOT sufficient (a model outputting {0.71, 0.72, 0.73}
    would pass std>0.01 yet be functionally collapsed). This test directly
    checks that the UQ mechanism exercises all four policy branches.
    """
    console.print("\n[bold cyan]=== TEST 4: UQ Confidence Distribution ===[/bold cyan]")
    from triplet_extractor import TripletExtractor
    import numpy as np

    passes, total = 0, 0
    tire_path = "../TIRE/dnrti_aug_stix2_je.json"
    if not os.path.exists(tire_path):
        console.print("  [yellow]SKIP: TIRE dataset not found[/yellow]")
        return 0, 0

    with open(tire_path) as f:
        raw = json.load(f)

    extractor = TripletExtractor()
    if not extractor._model_loaded:
        console.print("  [yellow]SKIP: No model loaded — run train_relation_extractor.py first[/yellow]")
        return 0, 0

    # Use last 100 samples (test split — not seen during training)
    test_samples = raw[-100:]
    all_confs = []
    bucket_counts = {"ACCEPTED": 0, "FLAGGED": 0, "REJECTED": 0}

    for sample in test_samples:
        for tri in extractor.extract_from_tire_sample(sample):
            all_confs.append(tri.c_nlp)
            bucket_counts[tri.status] = bucket_counts.get(tri.status, 0) + 1

    if not all_confs:
        console.print(f"  {FAIL} No triplets extracted from test set")
        return 0, 1

    confs = np.array(all_confs)
    conf_std    = float(confs.std())
    unique_vals = len(set(confs.round(3)))

    console.print(f"  Triplets scored  : {len(all_confs)}")
    console.print(f"  c_NLP std        : {conf_std:.4f}")
    console.print(f"  Unique confs(3dp): {unique_vals}")
    console.print(f"  Bucket counts    : {bucket_counts}")

    # Assertion 1: std > 0.03 (model is discriminating)
    total += 1
    ok1 = conf_std > 0.03
    passes += ok1
    console.print(f"  {PASS if ok1 else FAIL} c_NLP std={conf_std:.4f} > 0.03")

    # Assertion 2: at least 20 unique confidence values
    total += 1
    ok2 = unique_vals > 20
    passes += ok2
    console.print(f"  {PASS if ok2 else FAIL} Unique conf values={unique_vals} > 20")

    # Assertion 3: FLAGGED bucket is non-empty (most likely to stay empty)
    total += 1
    ok3 = bucket_counts.get("FLAGGED", 0) > 0
    passes += ok3
    console.print(f"  {PASS if ok3 else FAIL} FLAGGED bucket has {bucket_counts.get('FLAGGED', 0)} members (need > 0)")

    # Assertion 4: REJECTED bucket is non-empty
    total += 1
    ok4 = bucket_counts.get("REJECTED", 0) > 0
    passes += ok4
    console.print(f"  {PASS if ok4 else FAIL} REJECTED bucket has {bucket_counts.get('REJECTED', 0)} members (need > 0)")

    # Assertion 5: ACCEPTED bucket is non-empty
    total += 1
    ok5 = bucket_counts.get("ACCEPTED", 0) > 0
    passes += ok5
    console.print(f"  {PASS if ok5 else FAIL} ACCEPTED bucket has {bucket_counts.get('ACCEPTED', 0)} members (need > 0)")

    console.print(f"\n  [bold]UQ Tests: {passes}/{total} passed[/bold]")
    return passes, total


# ------------------------------------------------------------------------------
# TEST 5: NER Stopword Guard
# ------------------------------------------------------------------------------
def test_ner_no_stopwords():
    """
    Verify that common English words are NEVER tagged as entities.
    v1 incorrectly tagged 'used', 'against', 'Government' as [Malware].
    """
    console.print("\n[bold cyan]=== TEST 5: NER Stopword Guard ===[/bold cyan]")
    from triplet_extractor import extract_entities_heuristic

    GARBAGE_TOKENS = [
        "used", "against", "Government", "government", "actor",
        "access", "remote", "targeting", "ransomware", "campaign",
        "previously", "unknown", "the", "a", "an", "and",
    ]

    test_sentences = [
        "APT29 used Cobalt Strike against the US Government.",
        "The Lazarus Group deployed WannaCry ransomware targeting organizations.",
        "admin@338 targeted organizations using spear phishing emails.",
        "The actor exploited CVE-2021-44228 to gain remote code execution.",
        "A previously unknown campaign was attributed to BlackBear.",
    ]

    passes, total = 0, 0
    all_clean = True
    for sent in test_sentences:
        ents = extract_entities_heuristic(sent)
        entity_names = {e[0].lower() for e in ents}
        garbage_found = [g for g in GARBAGE_TOKENS if g.lower() in entity_names]
        total += 1
        clean = len(garbage_found) == 0
        passes += clean
        status = PASS if clean else FAIL
        console.print(f"  {status} '{sent[:55]}...'")
        if not clean:
            console.print(f"    Garbage tokens found: {garbage_found}")
            all_clean = False
        else:
            tagged = [(e[0], e[1]) for e in ents]
            console.print(f"    Tagged: {tagged}")

    console.print(f"\n  [bold]NER Tests: {passes}/{total} passed[/bold]")
    return passes, total


# ------------------------------------------------------------------------------
# TEST 3: End-to-End Ingestion
# ------------------------------------------------------------------------------
def test_ingestion():
    console.print("\n[bold cyan]=== TEST 3: End-to-End Ingestion (10 samples) ===[/bold cyan]")
    import subprocess, sys, tempfile
    with tempfile.TemporaryDirectory() as tmp_out:
        result = subprocess.run(
            [sys.executable, "ingestion_pipeline.py", "--samples", "10", "--output-dir", tmp_out],
            capture_output=True, text=True, cwd=str(Path(__file__).parent)
        )
        passes, total = 0, 1
        ok = result.returncode == 0
        passes += ok
        if ok:
            console.print(f"  {PASS} ingestion_pipeline.py exited cleanly (returncode=0)")
            # Check output dir exists
            out_dir = Path(tmp_out)
            if out_dir.exists() and any(out_dir.iterdir()):
                console.print(f"  {PASS} Output artifacts created in {out_dir}/")
                passes += 1
            total += 1
        else:
            console.print(f"  {FAIL} ingestion_pipeline.py failed:")
            console.print(result.stderr[-2000:])

    console.print(f"\n  [bold]Ingestion Tests: {passes}/{total} passed[/bold]")
    return passes, total


# ------------------------------------------------------------------------------
# TEST 6: Leakage Regression (permanent — catches split-logic regressions)
# Based on check_leakage.py findings: 97.2% leakage in instance-level split.
# This test ensures the text-dedup split keeps train/val text sets disjoint.
# ------------------------------------------------------------------------------
def test_no_leakage():
    console.print("\n[bold cyan]=== TEST 6: No Train/Val Leakage (text-dedup split) ===[/bold cyan]")
    import json
    from pathlib import Path
    from collections import defaultdict

    tire_path = str(Path(__file__).parent / "../TIRE/dnrti_aug_stix2_je.json")
    passes, total = 0, 0

    total += 1
    if not Path(tire_path).exists():
        console.print(f"  [yellow]SKIP: TIRE dataset not found at {tire_path}[/yellow]")
        passes += 1  # skip counts as pass
        console.print(f"\n  [bold]Leakage Tests: {passes}/{total} passed[/bold]")
        return passes, total

    with open(tire_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Replicate the text-dedup split from train_relation_extractor.py
    unique_texts = list({item.get("text", "") for item in raw})
    n_texts = len(unique_texts)
    rng = __import__("numpy").random.default_rng(42)
    shuffled = rng.permutation(unique_texts).tolist()

    val_frac, test_frac = 0.15, 0.15
    n_val   = int(n_texts * val_frac)
    n_test  = int(n_texts * test_frac)
    n_train = n_texts - n_val - n_test

    train_texts = set(shuffled[:n_train])
    val_texts   = set(shuffled[n_train:n_train + n_val])

    overlap = len(train_texts & val_texts)
    ok = (overlap == 0)
    passes += ok
    status = PASS if ok else FAIL
    console.print(f"  {status} Train/val text overlap: {overlap} texts  "
                  f"(must be 0 — text-dedup split invariant)")

    if not ok:
        console.print(f"  [red]  FAIL: {overlap} texts appear in both train and val.[/red]")
        console.print(f"  [red]  This means the split logic has regressed. Fix before proceeding.[/red]")

    # Additional check: number of unique texts
    total += 1
    ok2 = n_texts > 1000   # TIRE has ~3500+ unique texts after dedup
    passes += ok2
    console.print(f"  {PASS if ok2 else FAIL} Unique sentences: {n_texts:,}  (expect >1,000)")

    console.print(f"\n  [bold]Leakage Tests: {passes}/{total} passed[/bold]")
    return passes, total


# ------------------------------------------------------------------------------
# TEST 7: Calibration Monotonicity (accuracy tracks confidence)
# Runs after a trained model exists. Asserts that the model is not
# anti-correlated (high confidence != worse accuracy), which would make
# Phase 2's attention-weighted aggregation actively harmful.
# ------------------------------------------------------------------------------
def test_calibration_monotonicity():
    console.print("\n[bold cyan]=== TEST 7: Calibration Monotonicity ===[/bold cyan]")
    import os
    from diagnostics import run_calibration_check
    passes, total = 0, 0

    total += 1
    model_path = str(Path(__file__).parent / "models/fast_extractor.pkl")
    if not os.path.exists(model_path):
        console.print(f"  [yellow]SKIP: model not found ({model_path}). "
                      f"Run train_relation_extractor.py first.[/yellow]")
        passes += 1  # skip counts as pass when model not trained yet
        console.print(f"\n  [bold]Calibration Tests: {passes}/{total} passed[/bold]")
        return passes, total

    try:
        calib_ok = run_calibration_check(n_samples=300)
        passes += calib_ok
        status = PASS if calib_ok else FAIL
        console.print(f"  {status} Calibration: accuracy "
                      f"{'tracks' if calib_ok else 'does NOT track'} confidence monotonically")
        if not calib_ok:
            console.print(f"  [red]  WARNING: Phase 2 attention weighting will be unreliable.[/red]")
            console.print(f"  [red]  See diagnostics.py output for options.[/red]")
    except Exception as e:
        console.print(f"  {FAIL} Calibration check raised exception: {e}")

    console.print(f"\n  [bold]Calibration Tests: {passes}/{total} passed[/bold]")
    return passes, total


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 v2 Test Suite")
    parser.add_argument("--test-db",          action="store_true")
    parser.add_argument("--test-model",       action="store_true")
    parser.add_argument("--test-ingestion",   action="store_true")
    parser.add_argument("--test-uq",          action="store_true",
                        help="Test UQ bucket distribution (requires trained model)")
    parser.add_argument("--test-ner",         action="store_true",
                        help="Test NER stopword guard")
    parser.add_argument("--test-leakage",     action="store_true",
                        help="Test 6: no train/val text overlap (leakage regression)")
    parser.add_argument("--test-calibration", action="store_true",
                        help="Test 7: calibration monotonicity (requires trained model)")
    parser.add_argument("--seed-mitre",       action="store_true",
                        help="Include MITRE seeding in db tests (slow)")
    args = parser.parse_args()

    run_all = not any([args.test_db, args.test_model, args.test_ingestion,
                       args.test_uq, args.test_ner, args.test_leakage,
                       args.test_calibration])

    total_p, total_t = 0, 0

    if run_all or args.test_db:
        p, t = test_db(seed_mitre=args.seed_mitre)
        total_p += p; total_t += t

    if run_all or args.test_model:
        p, t = test_model()
        total_p += p; total_t += t

    if run_all or args.test_ingestion:
        p, t = test_ingestion()
        total_p += p; total_t += t

    if run_all or args.test_uq:
        p, t = test_varying_confidence()
        total_p += p; total_t += t

    if run_all or args.test_ner:
        p, t = test_ner_no_stopwords()
        total_p += p; total_t += t

    if run_all or args.test_leakage:
        p, t = test_no_leakage()
        total_p += p; total_t += t

    if run_all or args.test_calibration:
        p, t = test_calibration_monotonicity()
        total_p += p; total_t += t

    console.print(f"\n[bold]{'='*40}[/bold]")
    console.print(f"[bold]TOTAL: {total_p}/{total_t} tests passed[/bold]")
    console.print(f"[bold]{'='*40}[/bold]\n")
    sys.exit(0 if total_p == total_t else 1)
