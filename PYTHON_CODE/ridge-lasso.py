import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/pineapple_test_oh.csv'  
data = pd.read_csv(file_path)

# Assume the dataset has columns 'Features_new' and 'Label_new'
# Replace 'Features_new' and 'Label_new' with the actual column names if they are different
Features_new = data[['Features_new']]
Label_new = data['Label_new']

# Convert 'Features_new' column to numeric
data['Features_new'] = pd.to_numeric(data['Features_new'], errors='coerce')

# Remove rows with non-numeric values
data = data.dropna(subset=['Features_new'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Features_new, Label_new, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha parameter
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Lasso Regression
lasso_model = Lasso(alpha=1.0)  # You can adjust the alpha parameter
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

# Evaluate the models
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f'Mean Squared Error (Ridge): {mse_ridge}')
print(f'Mean Squared Error (Lasso): {mse_lasso}')

# Plot the results
plt.scatter(X_test, y_test, color='black', label='Actual values')
plt.plot(X_test, y_pred_ridge, color='blue', linewidth=3, label='Ridge Regression')
plt.plot(X_test, y_pred_lasso, color='red', linewidth=3, label='Lasso Regression')
plt.title('Ridge and Lasso Regression')
plt.xlabel('Features_new')
plt.ylabel('Label_new')
plt.legend()
plt.show()
