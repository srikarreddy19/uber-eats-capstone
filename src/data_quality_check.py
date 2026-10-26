import pandas as pd
from pathlib import Path

for file in Path("data/raw").glob("*.csv"):
    df = pd.read_csv(file)
    print("\nDataset:", file.name)
    print("Rows and columns:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])