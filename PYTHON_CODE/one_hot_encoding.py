# Program for demonstration of one hot encoding 

# import libraries 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

# import the data required 
data = pd.read_csv('C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.csv/TEST_SET/PINEAPPLE_test.csv') 
#print(data.head()) 

#print(data['Label'].unique()) 
#print(data['Features'].unique()) 

#print(data['Label'].value_counts())
#print(data['Features'].value_counts())

# Converting type of columns to category 
data['Label'] = data['Label'].astype('category') 
data['Features'] = data['Features'].astype('category') 

# Assigning numerical values and storing it in another columns 
data['Label_new'] = data['Label'].cat.codes 
data['Features_new'] = data['Features'].cat.codes

# Create an instance of One-hot-encoder 
enc = OneHotEncoder() 

# Passing encoded columns 
  
enc_data = pd.DataFrame(enc.fit_transform( 
    data[['Label_new', 'Features_new']]).toarray()) 
  
# Merge with main 
New_df = data.join(enc_data) 
  
#print(New_df)
data.to_csv('pineapple_test_oh.csv',index=False)