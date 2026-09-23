import os
import json
import pandas as pd
from collections import Counter

print("=== DEEP DATASET INSPECTION AND RELEVANCE ANALYSIS ===")

# 1. Inspect MITRE ATT&CK JSON
mitre_path = "enterprise-attack.json"
if os.path.exists(mitre_path):
    print("\n--- 1. MITRE ATT&CK Dataset Summary ---")
    with open(mitre_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    objects = data.get("objects", [])
    print(f"Total objects: {len(objects)}")
    
    types = [obj.get("type") for obj in objects]
    type_counts = Counter(types)
    print("\nTop Object Types in MITRE ATT&CK:")
    for t, c in type_counts.most_common(10):
        print(f"  - {t}: {c}")
        
    # Check relationships
    relationships = [obj for obj in objects if obj.get("type") == "relationship"]
    print(f"\nTotal Relationship objects: {len(relationships)}")
    rel_types = [rel.get("relationship_type") for rel in relationships]
    rel_type_counts = Counter(rel_types)
    print("\nTop Relationship Types:")
    for rt, c in rel_type_counts.most_common(10):
        print(f"  - {rt}: {c}")
else:
    print("MITRE ATT&CK file not found.")

# 2. Inspect CyberNER
cyberner_path = os.path.join("CyberNER", "dataset", "cyberner_combined_stix.csv")
if os.path.exists(cyberner_path):
    print("\n--- 2. CyberNER Dataset Summary ---")
    df = pd.read_csv(cyberner_path)
    print(f"Total rows: {len(df)}")
    print("Columns:", df.columns.tolist())
    
    # Analyze tags
    stix_tags = df['STIX_Tag'].dropna()
    # Filter to get only the entity names (ignoring O, B-, I- prefixes)
    entities = [tag.split('-', 1)[1] for tag in stix_tags if '-' in tag]
    entity_counts = Counter(entities)
    print("\nEntity Label Distribution in CyberNER (Entity Level):")
    for entity, count in entity_counts.most_common(15):
        print(f"  - {entity}: {count}")
        
    print("\nUnique sources in CyberNER:")
    print(df['Source'].value_counts())
else:
    print("CyberNER dataset file not found.")

# 3. Inspect TIRE
tire_path = os.path.join("TIRE", "dnrti_aug_stix2_je.json")
if os.path.exists(tire_path):
    print("\n--- 3. TIRE Dataset Summary ---")
    with open(tire_path, 'r', encoding='utf-8') as f:
        tire_data = json.load(f)
    print(f"Total samples: {len(tire_data)}")
    
    # Collect all entities and relation types
    all_ent_labels = []
    all_relations = []
    
    for item in tire_data:
        # Collect labels
        ent_labels = item.get("ent_labels", [])
        all_ent_labels.extend(ent_labels)
        
        # Collect relations
        relations = item.get("relations", [])
        for rel in relations:
            # rel format usually [idx1, idx2, relation_type] or similar
            if len(rel) >= 3:
                all_relations.append(rel[0])
                
    print("\nEntity Label Distribution in TIRE:")
    for label, count in Counter(all_ent_labels).most_common(10):
        print(f"  - {label}: {count}")
        
    print("\nRelation Type Distribution in TIRE:")
    for rel, count in Counter(all_relations).most_common(15):
        print(f"  - {rel}: {count}")
        
    # Sample entry preview
    print("\nSample Item from TIRE:")
    sample = tire_data[0]
    print(json.dumps({
        "text": sample.get("text")[:150] + "...",
        "entities": sample.get("entities"),
        "relations": sample.get("relations"),
        "ent_labels": sample.get("ent_labels")
    }, indent=2))
else:
    print("TIRE dataset file not found.")
