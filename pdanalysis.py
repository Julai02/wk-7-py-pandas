# wk-7-py-pandas: Iris Dataset Analysis and Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
try:
    df = pd.read_csv('iris.csv')
    print("✅ Dataset loaded successfully.\n")
except FileNotFoundError:
    print("❌ File not found. Please ensure 'iris.csv' is in your working directory.")
    exit()

# Display first few rows
print("🔍 First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\n📊 Data types:")
print(df.dtypes)

print("\n🧼 Missing values:")
print(df.isnull().sum())

# Clean dataset (drop rows with missing values)
df.dropna(inplace=True)

# Task 2: Basic Data Analysis
print("\n📈 Basic statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
print("\n📊 Mean values grouped by species:")
grouped = df.groupby('species').mean()
print(grouped)

# Task 3: Data Visualization
sns.set(style="whitegrid")

# Line chart (simulated index-based trend)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['sepal_length'], label='Sepal Length')
plt.plot(df.index, df['petal_length'], label='Petal Length')
plt.title('Line Chart: Sepal vs Petal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(6, 4))
grouped['petal_length'].plot(kind='bar', color='skyblue')
plt.title('Bar Chart: Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.tight_layout()
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(6, 4))
plt.hist(df['sepal_width'], bins=15, color='lightgreen', edgecolor='black')
plt.title('Histogram: Sepal Width Distribution')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# Observations
print("\n🔍 Observations:")
print("- Iris-setosa tends to have shorter petal lengths compared to other species.")
print("- Sepal length and petal length show a positive correlation.")
print("- Sepal width has a fairly normal distribution.")
print("- Line chart shows consistent variation across the dataset index.")

