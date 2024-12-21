import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the data
data = pd.read_csv(r"E:\EDGE\1970-2023 Oil spill .csv") # Update this with your actual file path

# Extract categorical columns from the dataframe
# Assuming the dataset contains some categorical columns to encode
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Apply one-hot encoding to the categorical columns
if categorical_columns:
    one_hot_encoded = encoder.fit_transform(data[categorical_columns])

    # Create a DataFrame with the one-hot encoded columns
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the one-hot encoded dataframe with the original dataframe
    data_encoded = pd.concat([data, one_hot_df], axis=1)

    # Drop the original categorical columns
    data_encoded = data_encoded.drop(categorical_columns, axis=1)

    print(f"Encoded Data: \n{data_encoded}")
else:
    print("No categorical columns found in the dataset.")
