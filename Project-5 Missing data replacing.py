import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv(r"E:\EDGE\1970-2023 Oil spill .csv")  # Update this with your actual file path

# Inspect the dataset to identify features and target
print("Dataset Columns:", data.columns)
print("First few rows of the dataset:")
print(data.head())

# Extract features (independent variables) and target (dependent variable)
# Assuming 'Year' and 'Million Dollars' are columns in the dataset
if 'Year' in data.columns and 'Million Dollars' in data.columns:
    X = data[['Year']].values  # Feature: Year
    y = data[['Million Dollars']].values  # Target: Million Dollars

    # Handle missing data by replacing NaN with the mean for Million Dollars
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    y = imp.fit_transform(y)

    # Encode the Year column if necessary (e.g., if it contains non-numeric data)
    if X.dtype.kind in 'O':  # Check if the Year column is non-numeric
        lb = LabelEncoder()
        X[:, 0] = lb.fit_transform(X[:, 0])

    # Output the results
    print("Processed Features (X - Year):\n", X)
    print("\nProcessed Dependent Variable (y - Million Dollars):\n", y)

    # Plot Year vs Million Dollars
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.title('Year vs Million Dollars')
    plt.xlabel('Year')
    plt.ylabel('Million Dollars')
    plt.grid()
    plt.legend()
    plt.show()
else:
    print("Error: Required columns 'Year' and 'Million Dollars' are missing from the dataset.")
