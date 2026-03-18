import torch
from torch_geometric.data import Data
import RNA

# ── Featurizer functions ──────────────────────────
NUCLEOTIDES = ['A', 'U', 'G', 'C']
PURINES = {'A', 'G'}
GC_SET = {'G', 'C'}

def get_node_features(sequence, dot_bracket):
    N = len(sequence)
    features = []
    for i, (nuc, symbol) in enumerate(zip(sequence, dot_bracket)):
        node = [0.0] * 4
        nuc = nuc.upper()
        if nuc in NUCLEOTIDES:
            node[NUCLEOTIDES.index(nuc)] = 1.0
        node.append(1.0 if nuc in PURINES else 0.0)
        node.append(1.0 if nuc in GC_SET else 0.0)
        node.append(i / max(N - 1, 1))
        node.append(0.0 if symbol == '.' else 1.0)
        features.append(node)
    return features

def parse_base_pairs(dot_bracket):
    pairs = []
    stack = []
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return pairs

# ── Graph Builder ─────────────────────────────────
def sequence_to_graph(sequence, label=None):
    sequence = sequence.upper().replace('T', 'U')
    dot_bracket, mfe = RNA.fold(sequence)

    node_features = get_node_features(sequence, dot_bracket)
    x = torch.tensor(node_features, dtype=torch.float32)

    src, dst = [], []
    N = len(sequence)

    for i in range(N - 1):
        src.append(i);     dst.append(i + 1)
        src.append(i + 1); dst.append(i)

    for (i, j) in parse_base_pairs(dot_bracket):
        src.append(i); dst.append(j)
        src.append(j); dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.mfe = mfe
    data.seq = sequence

    if label is not None:
        data.y = torch.tensor([float(label)], dtype=torch.float32)

    return data

# ── Test ──────────────────────────────────────────
seq = "GGGAAACCC"
graph = sequence_to_graph(seq, label=85.0)

print(f"Sequence: {seq}")
print(f"Nodes:    {graph.x.shape}")
print(f"Edges:    {graph.edge_index.shape}")
print(f"Label:    {graph.y}")
print(f"MFE:      {graph.mfe:.2f} kcal/mol")