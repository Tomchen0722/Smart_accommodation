import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Check reviews.csv columns and shape
print("=== reviews.csv ===")
df = pd.read_csv('reviews.csv', nrows=5)
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print()
print(df.dtypes)
print()
print(df.head(3).to_string())
print()

# Check calendar.csv columns 
print("=== calendar.csv (first 3 rows) ===")
df_cal = pd.read_csv('calendar.csv', nrows=3)
print(f"Columns ({len(df_cal.columns)}): {list(df_cal.columns)}")
print(df_cal.head(3).to_string())
print()

# Get listings full shape
print("=== listings.csv full shape ===")
df_l = pd.read_csv('listings.csv')
print(f"Shape: {df_l.shape}")
print(f"Missing values:\n{df_l.isnull().sum()}")
