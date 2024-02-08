import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Load your CSV dataset
data = pd.read_csv('C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/TEST_SET/PINEAPPLE_test.csv')

# Assuming 'Features' is the column containing string representations of arrays
data['Features'] = data['Features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Convert the 'Features' column to a NumPy array
X = np.vstack(data['Features'].values)

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Apply K-Means to cluster the data
num_clusters = 3  # You can choose the number of clusters based on your dataset
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X_normalized)

# Define num_classes outside the loop
num_classes = 10  # Replace with your actual number of classes

# Iterate over clusters and apply CNN
for cluster_id in range(num_clusters):
    # Extract data for the current cluster
    cluster_data = data[data['cluster'] == cluster_id]['Features'].values

    # Convert the 'Features' column to a NumPy array
    cluster_data = np.vstack(cluster_data)

    # Check if the cluster has enough samples for splitting
    if cluster_data.shape[0] < 2:
        print(f"Skipping cluster {cluster_id} due to insufficient samples.")
        continue

    # Split data into training and testing sets
    X_train, X_test = train_test_split(cluster_data, test_size=0.2, random_state=42)

    # Reshape data for CNN (assuming your features represent images)
    img_size = int(np.sqrt(X_train.shape[1]))  # Adjust this based on your data
    X_train = X_train.reshape(-1, img_size, img_size, 1)
    X_test = X_test.reshape(-1, img_size, img_size, 1)

    # Assuming you have a target variable y_train, y_test
    y_train = np.zeros((X_train.shape[0], num_classes))  # Replace this line with your actual target variable
    y_test = np.zeros((X_test.shape[0], num_classes))    # Replace this line with your actual target variable

    # Create a simple CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(img_size, img_size, 1)))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model (customize as needed)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

