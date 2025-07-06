
# Sales Prediction using Simple Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Advertising.csv")  # Ensure the file is in the same directory

# Display basic info
print("First 5 rows of the dataset:")
print(df.head())

print("\nChecking for null values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

# Visualize correlation
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot of TV vs Sales
sns.scatterplot(x='TV', y='Sales', data=df)
plt.title("TV Advertising Spend vs Sales")
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.show()

# Define feature and target
X = df[['TV']]
y = df['Sales']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Compare actual vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of Actual and Predicted Sales:")
print(comparison.head())

# Evaluate model
print("\nModel Evaluation:")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Plot regression line
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.title("Regression Line: TV vs Sales")
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.legend()
plt.show()
