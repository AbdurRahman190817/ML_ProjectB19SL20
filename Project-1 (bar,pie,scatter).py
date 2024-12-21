import csv
import numpy as np
import matplotlib.pyplot as plt

# File path to the uploaded CSV file
file_path = r"E:\EDGE\1970-2023 Oil spill .csv"

# Read the data from the CSV file
years = []
quantities = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header if it exists
    for row in reader:
        try:
            year = int(row[0].strip())  # Ensure the year is read as an integer
            quantity = float(row[1].strip())  # Ensure quantity is read as a float
            if 2000 <= year <= 2023:  # Filter for years 2013 to 2023
                years.append(year)
                quantities.append(quantity)
        except ValueError:
            print(f"Skipping invalid row: {row}")  # Handle non-numeric data gracefully

# Convert to NumPy arrays for convenience
years_array = np.array(years)
quantities_array = np.array(quantities)

# Ensure data alignment
if len(years_array) != len(quantities_array):
    print("Data mismatch between years and quantities.")
    exit()

# Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(years_array, quantities_array, color='skyblue', edgecolor='black')
plt.title('Global Quantity of Oil Spilled (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Quantity of Oil Spilled')
plt.xticks(years_array, rotation=45)  # Ensure all years are displayed correctly
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(quantities_array, labels=years_array, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
plt.title('Percentage of Oil Spills by Year (2000-2023)')
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(years_array, quantities_array, color='red', s=100, edgecolor='black', alpha=0.7)
plt.title('Scatter Plot of Oil Spills (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Quantity of Oil Spilled')
plt.xticks(years_array, rotation=45)  # Ensure all years are displayed correctly
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
