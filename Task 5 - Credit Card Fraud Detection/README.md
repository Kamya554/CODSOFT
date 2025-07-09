# Task 5 - Credit Card Fraud Detection

## 🔍 Objective
This project aims to build a machine learning model to detect fraudulent credit card transactions using supervised learning techniques.

## 📂 Dataset
The dataset used in this project is publicly available on Kaggle:

🔗 [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Download `creditcard.csv` and place it in the same directory as the Python script before running the code.

## 🧰 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest)
- Matplotlib, Seaborn

## 🛠️ Steps Performed
1. Loaded and explored the dataset.
2. Scaled the 'Amount' feature and dropped 'Time'.
3. Split the dataset into training and testing sets.
4. Trained Logistic Regression and Random Forest models.
5. Evaluated model performance using classification report and confusion matrix.

## 📈 Results
- Evaluated both models using accuracy, precision, recall, and F1-score.
- Plotted a confusion matrix for Random Forest predictions.

## 💻 Run the Code
1. Ensure `creditcard.csv` is in the same directory.
2. Run the script using:
   ```bash
   python credit_card_fraud_detection.py
