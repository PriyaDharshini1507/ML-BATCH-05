import os
import cv2
import pandas as pd
# Create an empty DataFrame
df = pd.DataFrame(columns=['Label', 'Features'])

image_folder = 'C:/Users/Priyadharshini/OneDrive/Desktop/ML(PROJECT)/DATASET/Dataset.img/Test_set/WATERMELON'

# Function to extract features from an image (replace with your own feature extraction logic)
def extract_features(image_path):
    img = cv2.imread(image_path)
    # Example: Resize image to a fixed size (you may need more complex preprocessing)
    img_resized = cv2.resize(img, (64, 64))
    # Example: Flatten image pixels as a feature vector
    features = img_resized.flatten()
    return features

# Function to extract label from filename
def get_label_from_filename(filename):
    # Assuming filenames are in the format "label_image123.jpg"
    label = filename.split('_')[0]
    return label

# Iterate over image files
# Iterate over image files
for filename in os.listdir(image_folder):
    filepath = os.path.join(image_folder, filename)
    
    # Extract features from the image
    features = extract_features(filepath)
    
    # Extract label from the filename
    label = get_label_from_filename(filename)
    
    # Create a new DataFrame for the current image
    df_row = pd.DataFrame({'Label': [label], 'Features': [features]})
    
    # Concatenate the new DataFrame with the main DataFrame
    df = pd.concat([df, df_row], ignore_index=True)
                    
# Save DataFrame to CSV
df.to_csv('WATERMELON_test.csv',index=False)