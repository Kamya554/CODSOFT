# Movie Rating Prediction with Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv('imdb_top_india.csv')  # Make sure the CSV file is in the same folder

# Step 2: Explore
print("First 5 rows:")
print(df.head())
print("Dataset info:")
print(df.info())

# Step 3: Clean missing data
df.dropna(inplace=True)

# Step 4: Encode categorical columns
categorical_cols = ['genre', 'director', 'actors']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

# Step 5: Features and target
X = df.drop('rating', axis=1)
y = df['rating']

# Step 6: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predict
y_pred = model.predict(X_test)

# Step 9: Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Step 10: Plot
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Movie Ratings')
plt.grid(True)
plt.show()
