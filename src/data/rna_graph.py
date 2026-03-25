# RNA Graph Builder - Version 2.0
# Updated for 12-dim node features
# AI4S Lab - Sazzad Hossain

import torch
from torch_geometric.data import Data
import RNA
from featurizer import get_node_features, compute_accessibility

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

def sequence_to_graph(sequence, label=None):
    sequence = sequence.upper().replace('T', 'U')
    dot_bracket, mfe = RNA.fold(sequence)

    # Accessibility compute করো
    accessibility = compute_accessibility(sequence)

    # Node features (12-dim)
    node_features = get_node_features(sequence, dot_bracket, accessibility)
    x = torch.tensor(node_features, dtype=torch.float32)

    # Edges
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