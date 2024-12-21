import csv
import numpy as np
import matplotlib.pyplot as plt

# File path to the uploaded CSV file
file_path = r"E:\EDGE\1970-2023 Oil spill .csv"
# Read the data from the CSV file
years = []
areas = []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header if it exists
    for row in reader:
        try:
            year = int(row[0].strip())  # Ensure the year is read as an integer
            area = float(row[2].strip())  # Assuming area coverage is the third column
            if 2000 <= year <= 2023:  # Filter for years 2013 to 2023
                years.append(year)
                areas.append(area)
        except ValueError:
            print(f"Skipping invalid row: {row}")  # Handle non-numeric data gracefully

# Convert to NumPy arrays for convenience
years_array = np.array(years)
areas_array = np.array(areas)

# Ensure data alignment
if len(years_array) != len(areas_array):
    print("Data mismatch between years and areas.")
    exit()

# Bar Chart
plt.figure(figsize=(10, 6))
plt.bar(years_array, areas_array, color='black', edgecolor='black')
plt.title('Global Area Coverage of Oil Spills (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Area Coverage (e.g., sq. km)')
plt.xticks(years_array, rotation=45)  # Ensure all years are displayed correctly
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(areas_array, labels=years_array, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
plt.title('Percentage of Oil Spill Area Coverage by Year (2000-2023)')
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(years_array, areas_array, color='blue', s=100, edgecolor='black', alpha=0.7)
plt.title('Scatter Plot of Oil Spill Area Coverage (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Area Coverage (sq. km)')
plt.xticks(years_array, rotation=45)  # Ensure all years are displayed correctly
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
