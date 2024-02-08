import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/TRAIN_SET/PINEAPPLE_train.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Display basic statistics of numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# Group by 'Label' and compute statistics for each group
grouped_stats = df.groupby('Label').describe()

# Display statistics for each label
print("\nGrouped Statistics:")
print(grouped_stats)

# Perform other statistical analyses as needed

# For example, you can compute correlations between features
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plotting coefficients

# Pairplot for pairwise relationships in the dataset
sns.pairplot(df, hue='Label')
plt.show()

# Correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()