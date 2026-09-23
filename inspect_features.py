import os
import json
import pandas as pd

print("==================================================")
print("     STAGE 1 DATASET FEATURE INSPECTION SYSTEM     ")
print("==================================================\n")

# ==================================================
# 1. INSPECTING DATASET 1: MITRE ATT&CK STIX JSON
# ==================================================
mitre_file = "enterprise-attack.json"

print("[+] Step 1: Parsing MITRE ATT&CK Features...")
if os.path.exists(mitre_file):
    with open(mitre_file, "r", encoding="utf-8") as f:
        mitre_data = json.load(f)
    
    # Extract the top-level keys
    print(f"Top-Level JSON Keys: {list(mitre_data.keys())}")
    
    # Dig into the objects to look at the entity attributes (features)
    objects = mitre_data.get("objects", [])
    print(f"Total Graph Objects Loaded: {len(objects)}")
    
    if objects:
        sample_obj = objects[0]
        print("\n--> Structural Features found inside a MITRE Graph Object:")
        print("-" * 65)
        for key, val in sample_obj.items():
            print(f"Feature: {key:<18} | Type: {type(val).__name__:<8} | Sample: {str(val)[:30]}")
        print("-" * 65)
else:
    print(f"[-] Missing: {mitre_file} not found in path.")

print("\n" + "="*50 + "\n")

# ==================================================
# 2. INSPECTING DATASET 2: CyberNER REPOSITORY
# ==================================================
print("[+] Step 2: Locating CyberNER Dataset Files...")
cyberner_dir = os.path.join("CyberNER", "dataset")

if os.path.exists(cyberner_dir):
    files = os.listdir(cyberner_dir)
    print(f"Available files inside CyberNER/dataset/: {files}")
    
    # Attempt to read a file to inspect schema features
    target_file = next((f for f in files if f.endswith('.json') or f.endswith('.csv') or f.endswith('.txt')), None)
    if target_file:
        full_path = os.path.join(cyberner_dir, target_file)
        print(f"\n--> Inspecting Features inside CyberNER sample file: '{target_file}'")
        print("-" * 65)
        if target_file.endswith('.json'):
            with open(full_path, 'r', encoding='utf-8') as f:
                c_data = json.load(f)
                if isinstance(c_data, list) and len(c_data) > 0:
                    print(f"JSON Block Features: {list(c_data[0].keys())}")
                elif isinstance(c_data, dict):
                    print(f"JSON Dict Features: {list(c_data.keys())}")
        else:
            # Display text lines if it uses sequential word token tags (BIO)
            with open(full_path, 'r', encoding='utf-8') as f:
                head = [f.readline().strip() for _ in range(5)]
            print("Linguistic Token Tags Format:")
            for line in head:
                print(f"  {line}")
        print("-" * 65)
else:
    print("[-] CyberNER/dataset directory not found. Make sure the git repo is intact.")

print("\n" + "="*50 + "\n")

# ==================================================
# 3. INSPECTING DATASET 3: TIRE RELATIONAL REPOSITORY
# ==================================================
print("[+] Step 3: Locating TIRE Relational Dataset Features...")
tire_dir = os.path.join("TIRE", "data")

# Fallback check if data is in the root directory of the repository
if not os.path.exists(tire_dir):
    tire_dir = "TIRE"

if os.path.exists(tire_dir):
    files = os.listdir(tire_dir)
    # Search for any json schema container
    json_files = [f for f in files if f.endswith('.json')]
    print(f"Available structured maps inside TIRE repo: {json_files}")
    
    if json_files:
        target_json = json_files[0]
        full_path = os.path.join(tire_dir, target_json)
        print(f"\n--> Inspecting Relational Features inside TIRE file: '{target_json}'")
        print("-" * 65)
        with open(full_path, 'r', encoding='utf-8') as f:
            t_data = json.load(f)
            if isinstance(t_data, list) and len(t_data) > 0:
                print(f"Relational Triple Matrix Keys: {list(t_data[0].keys())}")
                # Print sample text block schema if applicable
                print(f"Sample Entry Preview: {str(t_data[0])[:120]}...")
            elif isinstance(t_data, dict):
                print(f"Relational Schema Map Keys: {list(t_data.keys())}")
        print("-" * 65)
else:
    print("[-] TIRE directory not found.")

print("\nExecution complete! Share these structural console keys directly with your team.")