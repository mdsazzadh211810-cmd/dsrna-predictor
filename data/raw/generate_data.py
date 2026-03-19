# Large siRNA Dataset Generator
# Based on Huesken 2005 design rules
# AI4S Lab - Sazzad Hossain

import pandas as pd
import random

random.seed(42)

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

def calculate_efficiency(seq):
    """
    Huesken 2005 rules:
    1. GC content 30-52% = high efficiency
    2. A at position 19 = good
    3. U at position 1 = good
    4. No 4+ consecutive same nucleotide
    """
    score = 50.0

    # Rule 1: GC content
    gc = gc_content(seq)
    if 0.30 <= gc <= 0.52:
        score += 20
    elif gc < 0.25 or gc > 0.65:
        score -= 25

    # Rule 2: Position 19 = A
    if len(seq) >= 19 and seq[18] == 'A':
        score += 10

    # Rule 3: Position 1 = U or A
    if seq[0] in ['U', 'A']:
        score += 8

    # Rule 4: No 4+ same nucleotide
    for nuc in ['A', 'U', 'G', 'C']:
        if nuc * 4 in seq:
            score -= 20

    # Add biological noise
    noise = random.uniform(-8, 8)
    score += noise

    return max(5.0, min(98.0, score))

# Generate diverse sequences
nucleotides = ['A', 'U', 'G', 'C']

data = []
for i in range(500):
    # Random 21-mer siRNA
    seq = ''.join(random.choices(nucleotides, k=19))
    seq = seq + 'UU'  # siRNA always ends in UU

    efficiency = calculate_efficiency(seq)
    data.append((seq, round(efficiency, 1)))

# Save
df = pd.DataFrame(data, columns=['sequence', 'knockdown_efficiency'])
df.to_csv('data/raw/sirna_data.csv', index=False)

print(f"Total data: {len(df)} sequences")
print(f"\nDistribution:")
print(f"  High (>70%):   {len(df[df.knockdown_efficiency > 70])}")
print(f"  Medium (40-70%): {len(df[(df.knockdown_efficiency >= 40) & (df.knockdown_efficiency <= 70)])}")
print(f"  Low (<40%):    {len(df[df.knockdown_efficiency < 40])}")
print(f"\nSample data:")
print(df.head(5))