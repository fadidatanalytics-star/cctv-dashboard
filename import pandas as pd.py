import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a DataFrame
df = pd.read_excel("D:\Fadi\Projects_websites-Caridor\Projects\Coded - CCtv\SurveillanceCameras_latest-update2.xlsx")

# Display dataset info
print("Dataset Shape:", df.shape)
print("\n" + "="*80)
print("Column Names and Types:")
print("="*80)
print(df.dtypes)
print("\n" + "="*80)
print("First 10 rows:")
print("="*80)
print(df.head(10))
print("\n" + "="*80)
print("Dataset Summary Statistics:")
print("="*80)
print(df.describe())