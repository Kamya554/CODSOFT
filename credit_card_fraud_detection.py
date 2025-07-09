# Credit Card Fraud Detection using Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv("creditcard.csv")

# Step 2: Data exploration
print("Dataset Head:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nClass distribution:\n", df['Class'].value_counts())

# Step 3: Data preprocessing
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df = df.drop(['Time'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Step 5: Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Step 6: Evaluation

print("\nLogistic Regression Classification Report:\n")
print(classification_report(y_test, lr_preds))

print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, rf_preds))

# Step 7: Confusion Matrix (for Random Forest)
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
