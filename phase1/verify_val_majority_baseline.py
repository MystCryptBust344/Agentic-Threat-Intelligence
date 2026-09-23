"""
verify_val_majority_baseline.py
Confirms the 66.6% binary majority baseline on the val split is real
split variance, not a bug. Also prints exact text counts for guard assertions.
"""
import json
import numpy as np
from collections import defaultdict

with open("../TIRE/dnrti_aug_stix2_je.json", encoding="utf-8") as f:
    data = json.load(f)

# Exact split logic (3 bugs fixed — matches train_relation_extractor.py)
text_to_rows = defaultdict(list)
for d in data:
    text = d.get("text", "")
    ents = d.get("entities", [])
    for rel in d.get("relations", []):
        rt, i1, i2 = rel[0], rel[1], rel[2]
        if i1 >= len(ents) or i2 >= len(ents):
            continue
        text_to_rows[text].append({
            "binary": 1 if rt != "noRelation" else 0,
            "relation": rt
        })

unique_texts = list(text_to_rows.keys())
rng = np.random.default_rng(42)
shuffled = rng.permutation(unique_texts).tolist()
n = len(shuffled)
n_val  = int(n * 0.15)
n_test = int(n * 0.15)
n_train = n - n_val - n_test

texts_train = set(shuffled[:n_train])
texts_val   = set(shuffled[n_train:n_train + n_val])
texts_test  = set(shuffled[n_train + n_val:])

train_rows, val_rows, test_rows = [], [], []
for text, rows in text_to_rows.items():
    if text in texts_train:
        train_rows.extend(rows)
    elif text in texts_val:
        val_rows.extend(rows)
    else:
        test_rows.extend(rows)

print("=" * 60)
print("  Split Size Verification")
print("=" * 60)
print(f"  Train instances  : {len(train_rows):,}  (expected 43,996)")
print(f"  Val   instances  : {len(val_rows):,}  (expected  8,519)")
print(f"  Test  instances  : {len(test_rows):,}")
print(f"  Train text count : {len(texts_train)}  (expected 3780)")
print(f"  Val   text count : {len(texts_val)}    (expected  809)")

print("\n" + "=" * 60)
print("  Binary Val Split Distribution (sanity check)")
print("=" * 60)
y_val  = [r["binary"] for r in val_rows]
pos    = sum(y_val)
neg    = len(y_val) - pos
maj_acc = max(pos, neg) / len(y_val)
maj_cls = "has_relation" if pos >= neg else "noRelation"
print(f"  Val has_relation (1) : {pos:,}  ({100*pos/len(y_val):.2f}%)")
print(f"  Val noRelation   (0) : {neg:,}  ({100*neg/len(y_val):.2f}%)")
print(f"  Majority class        : {maj_cls}")
print(f"  Majority baseline acc : {maj_acc:.4f}  ({100*maj_acc:.2f}%)")
print(f"  Reported in script    : 66.64%")
ok = abs(maj_acc - 0.6664) < 0.001
print(f"  Match check           : {'PASS' if ok else 'FAIL — investigate'}")

print("\n" + "=" * 60)
print("  Full Corpus Binary Distribution (reference)")
print("=" * 60)
all_rows = train_rows + val_rows + test_rows
pos_all = sum(r["binary"] for r in all_rows)
neg_all = len(all_rows) - pos_all
print(f"  Full corpus positive  : {pos_all:,}  ({100*pos_all/len(all_rows):.2f}%)")
print(f"  Full corpus noRelation: {neg_all:,}  ({100*neg_all/len(all_rows):.2f}%)")
print(f"  Val vs corpus pos gap : {100*pos/len(y_val) - 100*pos_all/len(all_rows):+.2f} pp")
print(f"  (Expected ~+6 pp split variance — plausible if val texts skew positive)")
