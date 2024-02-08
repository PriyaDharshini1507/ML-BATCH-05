import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/pineapple_test_oh.csv'  
data = pd.read_csv(file_path)

# Print the column names to check for the correct column name
print(data.columns)

# Assume the dataset has a column with the correct name
Features_new = data[['Features_new']]
Label_new = data['Label_new']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Features_new, Label_new, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the regression line
plt.scatter(X_test, y_test, color='black', label='Actual values')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression line')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show(block=True)  # Explicitly set block=True to make plt.show() blocking
print(data.columns)