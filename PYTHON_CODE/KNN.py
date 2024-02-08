import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/TEST_SET/PINEAPPLE_test.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Assuming the dataset has features in a column 'Features' and the target variable in 'Label'
features = data['Features']
target = data['Label']

# Convert string representations of pixel values to numeric values
features = features.apply(lambda x: np.fromstring(x[1:-1], sep=' ', dtype=int).tolist())

# Reshape the features to have one row per sample
features = np.vstack(features)

# Apply K-means clustering
n_clusters = 3  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
cluster_labels_train = kmeans.labels_

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Add cluster labels to the training set
X_train_with_clusters = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
X_train_with_clusters['Cluster'] = cluster_labels_train[:len(X_train)]

# Apply KNN classification on the training set with clusters
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors
knn.fit(X_train_with_clusters, y_train)

# Apply K-means clustering to the test set
cluster_labels_test = kmeans.predict(X_test)

# Add cluster labels to the test set
X_test_with_clusters = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
X_test_with_clusters['Cluster'] = cluster_labels_test

# Make predictions using KNN on the test set with clusters
y_pred = knn.predict(X_test_with_clusters)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Visualization
# 2D scatter plot of the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_train_with_clusters['feature_0'], X_train_with_clusters['feature_1'], c=X_train_with_clusters['Cluster'], cmap='viridis', label='Clusters')
plt.title('2D Scatter Plot of Clusters')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend()
plt.show()

# 3D scatter plot if you have more than 2 features
if X_train.shape[1] > 2:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train_with_clusters['feature_0'], X_train_with_clusters['feature_1'], X_train_with_clusters['feature_2'],
               c=X_train_with_clusters['Cluster'], cmap='viridis', label='Clusters')
    ax.set_title('3D Scatter Plot of Clusters')
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_zlabel('Feature 2')
    ax.legend()
    plt.show()