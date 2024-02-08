import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Corrected file path
data = pd.read_csv('C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/TEST_SET/PINEAPPLE_test.csv')

# Print column names to check for typos or case sensitivity
print(data.columns)

# Replace with the actual column names from your dataset
feature_column = 'Features'
target_column = 'Label'

# Display a sample of your data to verify column names
print(data.head())

# Drop rows with missing values in the specified columns
data.dropna(subset=[feature_column, target_column], inplace=True)

# Assuming 'Features' column contains arrays, convert them to strings
data['Features'] = data['Features'].apply(lambda x: ' '.join(map(str, np.fromstring(x[1:-1], sep=' '))))

# Encode categorical feature
le_feature = LabelEncoder()
data['feature_encoded'] = le_feature.fit_transform(data[feature_column])

# Standardize the encoded feature
scaler = StandardScaler()
data['feature_encoded'] = scaler.fit_transform(data[['feature_encoded']])

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(data[target_column])
y = to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['feature_encoded']], y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(32, input_shape=(1,), activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate and print metrics
accuracy = accuracy_score(y_test_classes, y_pred_classes)
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
classification_report_str = classification_report(y_test_classes, y_pred_classes)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report_str)

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
