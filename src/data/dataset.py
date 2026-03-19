# Dataset Loader
# AI4S Lab - Sazzad Hossain

import pandas as pd
from rna_graph import sequence_to_graph

def load_dataset(csv_path):
    """Read data from CSV file and create graphs"""
    
    # Step 1: Read CSV
    df = pd.read_csv(csv_path)
    print(f"Total data: {len(df)} sequences")
    
    # Step 2: Convert each sequence to a graph
    graphs = []
    for i, row in df.iterrows():
        seq = row['sequence']
        label = row['knockdown_efficiency']
        
        graph = sequence_to_graph(seq, label=label)
        graphs.append(graph)
        print(f"{i+1}. {seq[:10]}... → {label}%")
    
    return graphs

# Test
graphs = load_dataset("data/raw/sirna_data.csv")
print(f"\nTotal graphs: {len(graphs)}")
print(f"First graph nodes: {graphs[0].x.shape}")
print(f"First graph label: {graphs[0].y}")