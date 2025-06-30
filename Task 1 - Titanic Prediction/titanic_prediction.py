import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the data
df = pd.read_csv('train.csv')

# Step 2: Drop columns not needed
df_model = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Step 3: Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
df_model[["Age", "Embarked"]] = imputer.fit_transform(df_model[["Age", "Embarked"]])

# Step 4: Convert text columns to numbers
le = LabelEncoder()
df_model["Sex"] = le.fit_transform(df_model["Sex"])
df_model["Embarked"] = le.fit_transform(df_model["Embarked"])

# Step 5: Split data into features (X) and target (y)
X = df_model.drop("Survived", axis=1)
y = df_model["Survived"]

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Show results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
