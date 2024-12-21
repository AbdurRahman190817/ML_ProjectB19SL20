import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Load the dataset
file_path = r"E:\EDGE\1970-2023 Oil spill .csv"
data = pd.read_csv(file_path)

# Select features and target variable
# Assuming 'year' is the first column and 'million dollars' is the third column
X = data.iloc[:, [0, 2]].values  # Features: year and million dollars
y = (data.iloc[:, 2] > 50).astype(int).values  # Create a binary target: e.g., high/low spill cost

# Filter for years 2013 to 2023
filter_mask = (X[:, 0] >= 1970) & (X[:, 0] <= 2023)
X = X[filter_mask]
y = y[filter_mask]

# Splitting the dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
X_test[:, 1:] = scaler.transform(X_test[:, 1:])

# Fit the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Predicted Results:\n", y_pred)
print("\nAccuracy of the Model:", accuracy)

# Visualize the Decision Boundary
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Naive Bayes Classification: Year vs Million Dollars')
plt.xlabel('Year')
plt.ylabel('Million Dollars')
plt.legend()
plt.show()
