import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import skfuzzy as fuzz

# Load your CSV dataset
df = pd.read_csv('C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/pineapple_test_oh.csv')

# Convert string representations of arrays to actual arrays
df['features'] = df['label'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Extract the features (X) from the dataset
X = np.array(df['features'].tolist())

# Choose the number of clusters (c) and the number of neighbors (k)
c = 3
k = 5

# Use KNN to find the nearest neighbors for each data point
knn = NearestNeighbors(n_neighbors=k)
knn.fit(X)
distances, indices = knn.kneighbors(X)

# Create the membership matrix using the distances from KNN
membership_matrix = np.exp(-distances ** 2 / (2.0 * (np.std(distances) ** 2)))

# Apply Fuzzy C-Means clustering
cntr, u, _, _, _, _, _ = fuzz.cmeans(data=X.T, c=c, m=2, error=0.005, maxiter=1000, init=None)

# Assign each data point to the cluster with the highest membership value
clusters = np.argmax(u, axis=0)

# Add the cluster labels to the original DataFrame
df['Cluster'] = clusters

# Print the cluster centers
print("Cluster Centers:")
print(cntr)

# Display the DataFrame with cluster labels
print("\nData with Cluster Labels:")
print(df)
