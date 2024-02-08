import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/TRAIN_SET/WATERMELON_train.csv'
data = pd.read_csv(file_path)

# Specify the column names
features_column = 'Features'
label_column = 'Label'

# Convert values to numeric
data[features_column] = data[features_column].apply(lambda x: np.fromstring(str(x)[1:-1], sep=' ', dtype=float).tolist())

# Drop rows with NaN values in 'Features_new' column
data = data.dropna(subset=[features_column])

# Flatten the features for SVM
flattened_features = np.vstack(data[features_column]).astype(float)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(flattened_features, data[label_column], test_size=0.2, random_state=42)

# Train SVM on the training set
svm = SVC(kernel='linear')  # You can adjust the kernel and other parameters
svm.fit(X_train,y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Evaluate the SVM model
accuracy = accuracy_score(y_test, y_pred)

# Print information
print(f'Shape of X_train: {X_train.shape}')
print(f'Head of Features_new column:\n{data[features_column].head()}')
print(f'Accuracy: {accuracy}')

