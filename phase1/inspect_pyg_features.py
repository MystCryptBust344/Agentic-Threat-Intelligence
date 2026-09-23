import torch

path = "phase1_output/tkg_pyg_snapshot.pt"
data = torch.load(path, weights_only=False)

print("=" * 70)
print("  PyG Snapshot Feature Verification (Orthogonal Non-Collinear Features)")
print("=" * 70)
print(f"  PyG Data Object              : {data}")
print(f"  Node Count                   : {data.num_nodes:,}")
print(f"  Edge Count                   : {data.edge_index.shape[1]:,}")
print(f"  Edge Feature Tensor          : {data.edge_attr.shape} ({data.edge_attr.dtype})")
print("\n  First 10 Rows of edge_attr [c_nlp, source_reputation, normalized_corroboration]:")
print("  " + "-" * 65)
for i in range(min(10, data.edge_attr.shape[0])):
    c_nlp, rep, corrob = data.edge_attr[i].tolist()
    print(f"  Row {i:>2}: c_nlp={c_nlp:.4f} | source_reputation={rep:.2f} | norm_corroboration={corrob:.4f}")

print("\n  Column Summary Stats:")
print(f"    Col 0 (c_nlp)                  : min={data.edge_attr[:, 0].min():.4f}, mean={data.edge_attr[:, 0].mean():.4f}, max={data.edge_attr[:, 0].max():.4f}")
print(f"    Col 1 (source_reputation)      : min={data.edge_attr[:, 1].min():.4f}, mean={data.edge_attr[:, 1].mean():.4f}, max={data.edge_attr[:, 1].max():.4f}")
print(f"    Col 2 (norm_corroboration)     : min={data.edge_attr[:, 2].min():.4f}, mean={data.edge_attr[:, 2].mean():.4f}, max={data.edge_attr[:, 2].max():.4f}")
print("=" * 70)
