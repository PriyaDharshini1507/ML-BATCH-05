import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import scikitplot as skplt

def importdata():
    balance_data = pd.read_csv(r'C:\Users\Priyadharshini\OneDrive\Desktop\ML(PROJECT)\DATASET\Dataset.csv\TRAIN_SET/PINEAPPLE_train.csv', sep=',', header=None)

    # Printing the dataset shape
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)

    # Printing the dataset observations
    print("Dataset: ", balance_data.head())
    return balance_data

# Function to preprocess data and split the dataset
def preprocess_and_split(balance_data):
    # Separate the header and data
    header = balance_data.iloc[0]
    balance_data = balance_data[1:]
    balance_data.columns = header

    # Separate features and target variable
    X = balance_data.iloc[:, :-1]  # Features
    y = balance_data.iloc[:, -1]   # Target variable

    # Convert target variable to numerical using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Convert features to numerical using one-hot encoding
    label_encoder = LabelEncoder()
    X_encoded = X.copy()
    X_encoded['Label'] = label_encoder.fit_transform(X['Label'])

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.3, random_state=100)

    return X_train, X_test, y_train, y_test

# Function to perform training with Gini Index.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to make predictions
def prediction(X_test, clf_object):
    # Prediction on test
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy and plot confusion matrix
def cal_accuracy(y_test, y_pred, clf_object):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print("Report : ", classification_report(y_test, y_pred))


    # Plot confusion matrix using seaborn
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf_object.classes_, yticklabels=clf_object.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Plot decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf_object, feature_names=X_test.columns, class_names=np.unique(y_test).astype(str), filled=True)
    plt.show()

def main():
    # Building Phase
    data = importdata()
    X_train, X_test, y_train, y_test = preprocess_and_split(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)

    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using Gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini, clf_gini)

# Calling main function
if __name__ == "__main__":
    main()