# RNA Nucleotide Featurizer
# AI4S Lab, Guizhou University
# Developer: Sazzad Hossain

NUCLEOTIDES = ['A', 'U', 'G', 'C']
PURINES = {'A', 'G'}      # দুটো ring
PYRIMIDINES = {'U', 'C'}  # একটা ring
GC_SET = {'G', 'C'}       # বেশি stable

def nucleotide_one_hot(nuc):
    """A, U, G, C কে সংখ্যায় রূপান্তর"""
    vec = [0.0, 0.0, 0.0, 0.0]
    nuc = nuc.upper()
    if nuc in NUCLEOTIDES:
        vec[NUCLEOTIDES.index(nuc)] = 1.0
    return vec

def is_purine(nuc):
    """Purine কিনা"""
    return 1.0 if nuc.upper() in PURINES else 0.0

def is_gc(nuc):
    """GC pair কিনা - বেশি stable"""
    return 1.0 if nuc.upper() in GC_SET else 0.0

def get_node_features(sequence, dot_bracket):
    """
    RNA sequence থেকে সব features বের করো
    sequence: যেমন 'AUGCAUGC'
    dot_bracket: যেমন '((....))' - ViennaRNA থেকে
    """
    N = len(sequence)
    features = []

    for i, (nuc, symbol) in enumerate(zip(sequence, dot_bracket)):
        node = []
        node += nucleotide_one_hot(nuc)          # 4 features
        node.append(is_purine(nuc))               # 1 feature
        node.append(is_gc(nuc))                   # 1 feature
        node.append(i / max(N - 1, 1))            # position
        node.append(0.0 if symbol == '.' else 1.0) # paired?
        features.append(node)

    return features

# Test
seq = "GGGAAACCC"
struct = "(((...)))"
result = get_node_features(seq, struct)
print(f"Sequence length: {len(seq)}")
print(f"Feature per nucleotide: {len(result[0])}")
print(f"First nucleotide (G) features: {result[0]}")
print(f"Middle nucleotide (A) features: {result[3]}")