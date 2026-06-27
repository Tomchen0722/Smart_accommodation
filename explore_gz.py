import sys, gzip, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

with gzip.open('listings.csv.gz', 'rt', encoding='utf-8') as f:
    df = pd.read_csv(f, nrows=3)

print(f"Columns ({len(df.columns)}):")
for c in df.columns:
    print(f"  {c}: {df[c].dtype} | sample: {df[c].iloc[0]}")
