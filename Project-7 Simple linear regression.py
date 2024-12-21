import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv(r"E:\EDGE\1970-2023 Oil spill .csv")  # Path to the uploaded CSV file

# Display the dataset columns
print("Dataset Columns:", data.columns)

# Ensure the required columns exist
required_columns = ['Year', 'Quantity of Oil Spilled']
if all(column in data.columns for column in required_columns):
    # Extract features (Year) and target (Quantity of Oil Spilled)
    X = data[['Year']].values  # Feature: Year
    y = data['Quantity of Oil Spilled'].values  # Target: Quantity of Oil Spilled

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Fitting the training data using Linear Regression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting from the learned model
    y_predict = regressor.predict(X_test)

    print("\n The Original values=", y_test)
    print("\n The predicted values=", y_predict)

    # Visualize the data
    plt.scatter(X_train, y_train, color='green', label='Training Data')
    plt.plot(X_train, regressor.predict(X_train), color='red', label='Regression Line')
    plt.scatter(X_test, y_test, color='blue', label='Test Data')
    plt.title('Year vs Quantity of Oil Spilled')
    plt.xlabel('Year')
    plt.ylabel('Quantity of Oil Spilled')
    plt.legend()
    plt.show()
else:
    print("Error: Required columns 'Year' and 'Quantity of Oil Spilled' are missing from the dataset.")