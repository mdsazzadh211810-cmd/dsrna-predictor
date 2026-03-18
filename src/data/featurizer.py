# ViennaRNA দিয়ে real structure prediction
import RNA

def fold_rna(sequence):
    """RNA sequence fold করো এবং structure বের করো"""
    sequence = sequence.upper().replace('T', 'U')
    structure, mfe = RNA.fold(sequence)
    return structure, mfe

# Test
seq = "GGGAAACCC"
structure, energy = fold_rna(seq)
print(f"\nSequence:  {seq}")
print(f"Structure: {structure}")
print(f"Energy:    {energy} kcal/mol")