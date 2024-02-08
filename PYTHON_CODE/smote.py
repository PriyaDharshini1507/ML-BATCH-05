# import necessary modules 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split 
import ast

# load the data set 
data = pd.read_csv("C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/pineapple_test_oh.csv") 

# print info about columns in the dataframe 
print(data.info()) 
 
# Extract features and labels
def safe_eval(x):
    try:
        return np.array(ast.literal_eval(x)) if isinstance(x, str) else x
    except (SyntaxError, ValueError):
        # Handle the case where literal_eval fails
        return x

X = data['Features_new'].apply(safe_eval)
y = data['Label_new']

# split into 70:30 ration 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 

# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape) 

X_train_flat=np.vstack(X_train)
X_train_flat=np.array([item.flatten() for item in X_train_flat])

print("XTrain",X_train_flat)
print("YTain",y_train)
# logistic regression object 
lr = LogisticRegression() 

# train the model on train set 
lr.fit(X_train_flat, y_train) 

# reshape X_test to 2D array
X_test_flat = np.vstack(X_test)
X_test_flat = np.array([item.flatten() for item in X_test_flat])

# make predictions
predictions = lr.predict(X_test_flat)

# print classification report
print(classification_report(y_test, predictions))
