import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
file_path = r'E:\EDGE\1970-2023 Oil spill .csv'
data = pd.read_csv(file_path)

# Display the columns to understand the structure of the dataset
print("Dataset Columns:", data.columns)

# Ensure the required columns exist for processing
if 'Year' in data.columns and 'Quantity of Oil Spilled' in data.columns:
    # Extract features (Year and others if needed) and target (Quantity of Oil Spilled)
    X = data[['Year']].values  # Feature: Year
    y = data['Quantity of Oil Spilled'].values  # Target: Quantity of Oil Spilled

    # Encoding the feature if necessary (not needed here since 'Year' is numeric)
    # For demonstration, we assume additional encoding is not required.

    # Splitting the dataset into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("Training Features (X_train):", X_train)
    print("Testing Features (X_test):", X_test)

    # Standardizing the features
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)

    print("Scaled Training Features:", X_train_scaled)
    print("Scaled Testing Features:", X_test_scaled)
else:
    print("Error: Required columns 'Year' and 'Quantity of Oil Spilled' are missing from the dataset.")
