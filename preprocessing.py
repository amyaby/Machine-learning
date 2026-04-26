from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load the Iris dataset
iris = load_iris()

# --- Step 1: Split data ---
# Your code for splitting data goes here
X = iris.data
y = iris.target
print("Shape of features (X):", X.shape)
print("Shape of target (y):", y.shape)

# --- Step 2: Initialize the scaler ---
# Your code for creating a StandardScaler instance goes here
# --- Step 2: Initialize the scaler ---
scaler = StandardScaler()

print("Scaler object created:", scaler)

# --- Step 3: Fit the scaler ---
# Your code for fitting the scaler goes here
# --- Step 3: Fit the scaler ---
scaler.fit(X)

print("Scaler mean:", scaler.mean_)

# --- Step 4: Transform the data ---
# Your code for transforming the data goes here
# We will store the transformed data in a new variable, X_scaled, to keep the original data intact.
# --- Step 4: Transform the data ---
X_scaled = scaler.transform(X)

# Use numpy to set precision for cleaner output
np.set_printoptions(precision=2, suppress=True)
print("Original data mean:", np.mean(X, axis=0))
print("Scaled data mean:", np.mean(X_scaled, axis=0))
print("Scaled data sample:\n", X_scaled[:5])
# --- Step 5: Encode the target ---
# Your code for encoding the target variable goes here
# --- Step 5: Encode the target ---
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nOriginal target sample:", y[:5])  # Show first 5 original labels
print("Encoded target sample:", y_encoded[:5])  # Show first 5 encoded labels
print("Unique encoded values:", np.unique(y_encoded))  # Show all unique encoded values

