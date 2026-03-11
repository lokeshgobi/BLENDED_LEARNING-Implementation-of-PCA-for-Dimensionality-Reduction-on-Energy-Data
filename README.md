# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load the Heights and Weights dataset using Pandas and inspect the data.
2.Select the features (Height and Weight) and standardize them using StandardScaler.
3.Apply PCA (Principal Component Analysis) to reduce the features into two principal components.
4.Calculate explained variance and visualize the principal components using a scatter plot.
```
## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: LOKESHWARAN G
RegisterNumber:  212225040210
*/
```
```
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset from a local file
# Ensure the correct file path is provided where the dataset is saved
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/HeightsWeights.csv"
data = pd.read_csv(url)

# Step 2: Explore the data
# Display the first few rows and column names for initial inspection
print(data.head())
print(data.columns)

# Step 3: Preprocess the data (Feature Scaling)
# Select the relevant columns for analysis
X = data[['Height(Inches)', 'Weight(Pounds)']]  # Use the appropriate column names

# Standardize the features to bring them to the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA for dimensionality reduction
# Initialize PCA to reduce the features to 2 components (for simplicity)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Analyze the explained variance
# Print the explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio for each Principal Component:", explained_variance)
print("Total Explained Variance:", sum(explained_variance))

# Step 6: Visualize the principal components
# Create a DataFrame to store the principal components
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Plot the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Heights and Weights Dataset")
plt.show()
```
## Output:

<img width="981" height="626" alt="Screenshot 2026-03-10 195326" src="https://github.com/user-attachments/assets/7de347cf-cef2-441b-8d96-4f069cc144c8" />


## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
