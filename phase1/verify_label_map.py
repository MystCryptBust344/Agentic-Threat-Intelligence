"""
verify_label_map.py
===================
Verifies consistency between the fast_extractor.pkl model's internal classes_ array
and the label_map.json used by downstream extraction logic.
"""

import json
import pickle
import os

def check_label_map_consistency(model_path="models/fast_extractor.pkl", json_path="models/label_map.json"):
    if not os.path.exists(model_path) or not os.path.exists(json_path):
        print(f"ERROR: Missing model artifacts. Run train_relation_extractor.py first.")
        return

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # The model dict in Phase 1 saves 'classes' array explicitly alongside 'model' Pipeline
    # If the classes are directly on the dictionary:
    if isinstance(model, dict) and 'classes' in model:
        model_classes = list(model['classes'])
    elif hasattr(model, 'classes_'):
        model_classes = list(model.classes_)
    else:
        # try to get from pipeline
        try:
            model_classes = list(model.steps[-1][1].classes_)
        except Exception as e:
            print(f"ERROR: Could not extract classes_ array from pickle: {e}")
            return
            
    # Load JSON map
    with open(json_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
        
    if isinstance(label_map, dict) and "classes" in label_map:
        json_classes = label_map["classes"]
    elif isinstance(label_map, dict):
        # assume dict mapping label to index, sort by index
        json_classes = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    elif isinstance(label_map, list):
        json_classes = label_map
    else:
        print(f"ERROR: Unknown label_map.json structure: {type(label_map)}")
        return
        
    print("============================================================")
    print("  Label Map vs Model Classes Consistency Check")
    print("============================================================")
    
    match = (model_classes == json_classes)
    if match:
        print(f"[PASS] Model and JSON label maps match exactly ({len(model_classes)} classes).")
        print(f"       Order verification successful.")
    else:
        print("[FAIL] MISMATCH DETECTED!")
        print(f"Model classes: {model_classes}")
        print(f"JSON  classes: {json_classes}")

if __name__ == "__main__":
    check_label_map_consistency()
