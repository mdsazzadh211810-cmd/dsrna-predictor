# Enhanced RNA Featurizer - Version 2.0
# Added: thermodynamic features, accessibility score
# AI4S Lab - Sazzad Hossain

import RNA
import numpy as np

NUCLEOTIDES = ['A', 'U', 'G', 'C']
PURINES = {'A', 'G'}
GC_SET = {'G', 'C'}

# ── Feature 1-4: One-hot identity ─────────────────
def nucleotide_one_hot(nuc):
    vec = [0.0] * 4
    nuc = nuc.upper()
    if nuc in NUCLEOTIDES:
        vec[NUCLEOTIDES.index(nuc)] = 1.0
    return vec

# ── Feature 5: Purine indicator ───────────────────
def is_purine(nuc):
    return 1.0 if nuc.upper() in PURINES else 0.0

# ── Feature 6: GC indicator ───────────────────────
def is_gc(nuc):
    return 1.0 if nuc.upper() in GC_SET else 0.0

# ── Feature 7: Normalized position ───────────────
def normalized_position(i, N):
    return i / max(N - 1, 1)

# ── Feature 8: Paired status ──────────────────────
def paired_status(symbol):
    return 0.0 if symbol == '.' else 1.0

# ── Feature 9: Accessibility (NEW) ───────────────
def compute_accessibility(sequence):
    """
    ViennaRNA partition function দিয়ে
    প্রতিটি nucleotide-এর single-strandedness probability।
    Higher value = more accessible = easier for RISC to bind.
    """
    try:
        md = RNA.md()
        fc = RNA.fold_compound(sequence, md)
        fc.pf()
        bpp = fc.bpp()
        N = len(sequence)
        paired_prob = np.zeros(N)
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):
                p = bpp[i][j]
                paired_prob[i - 1] += p
                paired_prob[j - 1] += p
        accessibility = np.clip(1.0 - paired_prob, 0.0, 1.0)
        return accessibility
    except:
        return np.ones(len(sequence)) * 0.5

# ── Feature 10: Local GC content (NEW) ───────────
def local_gc_content(sequence, i, window=3):
    """
    Position i এর আশেপাশে ছোট window-এ GC content।
    Local thermodynamic stability measure।
    """
    start = max(0, i - window)
    end = min(len(sequence), i + window + 1)
    local_seq = sequence[start:end]
    gc = sum(1 for n in local_seq if n.upper() in GC_SET)
    return gc / len(local_seq) if local_seq else 0.0

# ── Feature 11: 5' end indicator (NEW) ───────────
def is_five_prime_end(i, N, window=3):
    """
    siRNA-এর 5' end (positions 1-3) RISC loading-এ গুরুত্বপূর্ণ।
    """
    return 1.0 if i < window else 0.0

# ── Feature 12: Seed region indicator (NEW) ──────
def is_seed_region(i):
    """
    Positions 2-8 হলো siRNA seed region।
    Target specificity নির্ধারণ করে।
    """
    return 1.0 if 1 <= i <= 7 else 0.0

# ── Main Feature Extractor ─────────────────────────
def get_node_features(sequence, dot_bracket, accessibility=None):
    """
    12-dimensional node features per nucleotide.
    
    Dimensions:
    [0-3]  One-hot nucleotide identity
    [4]    Purine indicator
    [5]    GC indicator
    [6]    Normalized position
    [7]    Paired status
    [8]    Accessibility (single-strandedness)
    [9]    Local GC content (window=3)
    [10]   5' end indicator
    [11]   Seed region indicator (positions 2-8)
    """
    N = len(sequence)

    if accessibility is None:
        accessibility = compute_accessibility(sequence)

    features = []
    for i, (nuc, symbol) in enumerate(zip(sequence, dot_bracket)):
        node = []
        node += nucleotide_one_hot(nuc)                    # [0-3]
        node.append(is_purine(nuc))                        # [4]
        node.append(is_gc(nuc))                            # [5]
        node.append(normalized_position(i, N))             # [6]
        node.append(paired_status(symbol))                 # [7]
        node.append(float(accessibility[i]))               # [8] NEW
        node.append(local_gc_content(sequence, i))        # [9] NEW
        node.append(is_five_prime_end(i, N))               # [10] NEW
        node.append(is_seed_region(i))                     # [11] NEW
        features.append(node)

    return features

NODE_FEATURE_DIM = 12

# Test
if __name__ == "__main__":
    seq = "GGGAAACCC"
    struct = "(((...)))"
    feats = get_node_features(seq, struct)
    print(f"Sequence: {seq}")
    print(f"Feature dimension: {len(feats[0])}")
    print(f"First nucleotide (G): {feats[0]}")
    print(f"Seed region (pos 2, A): {feats[2]}")