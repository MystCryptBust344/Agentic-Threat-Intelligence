"""
check_leakage.py - Investigates whether the 70/15/15 split has document-level leakage.

Since TIRE has no doc ID field, we use text similarity as a proxy:
- Exact duplicate texts across train/val
- Near-duplicate texts (edit distance / first-N-chars similarity)
- Whether top-N TF-IDF train vectors have near-identical val neighbors
"""
import json, collections, sys
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit

DATA_PATH = "../TIRE/dnrti_aug_stix2_je.json"

def main():
    print("=" * 60)
    print("  Leakage Analysis — TIRE dataset")
    print("=" * 60)

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"\nTotal samples: {len(data)}")
    print(f"Keys per sample: {list(data[0].keys())}")

    # ── 1. Duplicate text check ──────────────────────────────────
    texts = [d["text"] for d in data]
    text_counter = collections.Counter(texts)
    exact_dups = {t: c for t, c in text_counter.items() if c > 1}
    print(f"\nExact duplicate texts : {len(exact_dups)} groups "
          f"({sum(exact_dups.values())} instances)")

    prefix_counter = collections.Counter(t[:80] for t in texts)
    near_dups = {p: c for p, c in prefix_counter.items() if c > 1}
    print(f"Near-dups (first 80ch): {len(near_dups)} groups "
          f"({sum(near_dups.values())} instances)")

    if exact_dups:
        print("\nSample duplicate texts:")
        for txt, cnt in list(exact_dups.items())[:3]:
            print(f"  ({cnt}x) \"{txt[:100]}\"")

    # ── 2. Expand to (text, relation) instances (same as training) ─
    print("\n" + "-"*60)
    print("Expanding to (text, relation) instances...")
    instances, labels = [], []
    for sample in data:
        txt = sample["text"]
        for rel in sample["relations"]:
            instances.append(txt)
            labels.append(rel[0])

    print(f"Total instances: {len(instances)}")
    label_counts = collections.Counter(labels)
    print(f"Classes: {len(label_counts)}")
    print(f"Most common: {label_counts.most_common(3)}")

    # ── 3. Reproduce the train/val split ─────────────────────────
    from sklearn.model_selection import train_test_split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        instances, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15/0.85, random_state=42, stratify=y_trainval)

    print(f"\nSplit sizes → Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # ── 4. Check exact text overlap between train and val ─────────
    train_set = set(X_train)
    val_overlap = sum(1 for x in X_val if x in train_set)
    print(f"\nVal instances with EXACT match in train: {val_overlap} / {len(X_val)} "
          f"({100*val_overlap/len(X_val):.1f}%)")

    # ── 5. TF-IDF cosine similarity: nearest train neighbor per val instance
    print("\nComputing TF-IDF cosine similarity (train→val nearest neighbors)...")
    vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    # Sample to keep fast
    sample_size = min(1000, len(X_val))
    val_sample_idx = np.random.default_rng(0).choice(len(X_val), sample_size, replace=False)
    X_val_sample  = [X_val[i] for i in val_sample_idx]
    y_val_sample  = [y_val[i] for i in val_sample_idx]

    all_texts = X_train + X_val_sample
    tfidf_mat = vect.fit_transform(all_texts)
    train_mat = tfidf_mat[:len(X_train)]
    val_mat   = tfidf_mat[len(X_train):]

    # For each val instance, find max cosine sim to any train instance
    sim_matrix = cosine_similarity(val_mat, train_mat)  # (1000, n_train)
    max_sims   = sim_matrix.max(axis=1)
    best_idx   = sim_matrix.argmax(axis=1)

    # Bin the similarities
    thresholds = [0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    print("\n  Val→Train max cosine similarity distribution:")
    for lo, hi in zip([0] + thresholds[:-1], thresholds):
        n = ((max_sims >= lo) & (max_sims < hi)).sum()
        bar = "█" * int(40 * n / sample_size)
        print(f"  [{lo:.2f}-{hi:.2f}) {bar:40s} {100*n/sample_size:5.1f}% ({n})")
    n = (max_sims >= 1.0).sum()
    bar = "█" * int(40 * n / sample_size)
    print(f"  [1.00     ] {bar:40s} {100*n/sample_size:5.1f}% ({n})  ← exact copies")

    print(f"\n  Mean max-sim  : {max_sims.mean():.4f}")
    print(f"  Median max-sim: {np.median(max_sims):.4f}")
    print(f"  % sim > 0.95  : {100*(max_sims > 0.95).mean():.1f}%")
    print(f"  % sim = 1.00  : {100*(max_sims >= 1.0).mean():.1f}%")

    # ── 6. Show 5 highest-similarity val/train pairs ─────────────
    print("\n  Top 5 most similar val/train pairs:")
    top5 = np.argsort(max_sims)[-5:][::-1]
    for rank, vi in enumerate(top5, 1):
        ti = best_idx[vi]
        sim = max_sims[vi]
        vtext = X_val_sample[vi][:80]
        ttext = X_train[ti][:80]
        vrel  = y_val_sample[vi]
        trel  = y_train[ti]
        print(f"  [{rank}] sim={sim:.4f}  val_rel={vrel}  train_rel={trel}")
        print(f"      VAL  : \"{vtext}\"")
        print(f"      TRAIN: \"{ttext}\"")

    # ── 7. What happens to val_acc if we remove high-sim instances? ─
    print("\n" + "-"*60)
    print("  Estimating 'clean' val_acc (excluding sim>0.95 instances)...")
    try:
        import pickle
        with open("models/fast_extractor.pkl", "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        vectorizer = bundle["vectorizer"]

        all_val_preds = model.predict(vectorizer.transform(X_val))

        # Full val accuracy
        full_acc = (np.array(all_val_preds) == np.array(y_val)).mean()
        print(f"  Full val accuracy (all {len(X_val)} instances): {full_acc:.4f}")

        # On the sampled subset
        val_preds_sample = model.predict(vectorizer.transform(X_val_sample))
        # Mask: keep only low-similarity instances
        low_sim_mask = max_sims < 0.95
        if low_sim_mask.sum() > 0:
            low_sim_correct = (
                np.array(val_preds_sample)[low_sim_mask] ==
                np.array(y_val_sample)[low_sim_mask]
            ).mean()
            print(f"  Val accuracy (sim < 0.95, n={low_sim_mask.sum()}): {low_sim_correct:.4f}")
        high_sim_mask = max_sims >= 0.95
        if high_sim_mask.sum() > 0:
            high_sim_correct = (
                np.array(val_preds_sample)[high_sim_mask] ==
                np.array(y_val_sample)[high_sim_mask]
            ).mean()
            print(f"  Val accuracy (sim ≥ 0.95, n={high_sim_mask.sum()}): {high_sim_correct:.4f}")
    except Exception as e:
        print(f"  [SKIP] Could not load model: {e}")

    print("\n" + "="*60)
    print("  Leakage analysis complete.")
    print("="*60)

if __name__ == "__main__":
    main()
