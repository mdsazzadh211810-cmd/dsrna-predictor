# Real siRNA Data Downloader
# Source: Huesken et al. 2005 (Nature Biotechnology)
# AI4S Lab - Sazzad Hossain

import urllib.request
import pandas as pd
import os

print("Real siRNA data download just started...")

# Huesken 2005 dataset URL
url = "http://www.rnaiweb.com/RNAi/siRNA_Design/"

try:
    urllib.request.urlretrieve(url, "data/raw/huesken_raw.csv")
    print("Download finished!")
    
    # Check
    df = pd.read_csv("data/raw/huesken_raw.csv")
    print(f"total sequences: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3))

except Exception as e:
    print(f"Download failed: {e}")
    print("Need to do Manual download !")