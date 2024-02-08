import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
df = pd.read_csv("C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/TRAIN_SET/PINEAPPLE_train.csv")

# Replace 'YourFeaturesColumnName' with the actual column name containing arrays
feature_column_name = 'Features'
X = df[feature_column_name]


# Convert the string representations of arrays to actual NumPy arrays
df[feature_column_name] = df[feature_column_name].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Create a new DataFrame with the feature column expanded into separate columns
expanded_features = pd.DataFrame(df[feature_column_name].to_list(), columns=[f'Feature_{i}' for i in range(len(df[feature_column_name].iloc[0]))])

# Concatenate the new DataFrame with the original DataFrame
df = pd.concat([df, expanded_features], axis=1)

# Drop the original feature column
df = df.drop(feature_column_name, axis=1)

# Perform statistical feature analysis (example: correlation matrix)
correlation_matrix = df.corr()

# Print or use the correlation matrix as needed
print(correlation_matrix)

# Split the data into features (X) and labels (y)
X = df['Features']
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)

# Evaluate the accuracy and other metrics
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
