# RNA nucleotide features
NUCLEOTIDES = ['A', 'U', 'G', 'C']

def nucleotide_one_hot(nuc):
    vec = [0.0, 0.0, 0.0, 0.0]
    nuc = nuc.upper()
    if nuc in NUCLEOTIDES:
        idx = NUCLEOTIDES.index(nuc)
        vec[idx] = 1.0
    return vec

# Test
print(nucleotide_one_hot('A'))  # [1, 0, 0, 0]
print(nucleotide_one_hot('G'))  # [0, 0, 1, 0]
print(nucleotide_one_hot('U'))  # [0, 1, 0, 0]
print(nucleotide_one_hot('C'))  # [0, 0, 0, 1]