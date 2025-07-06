# 📊 Task 4: Sales Prediction Using Python

This project is part of my **Data Science Internship at CODSOFT**. The goal is to build a simple linear regression model to predict product sales based on TV advertising spend.

---

## 📝 Project Description

Sales prediction is essential in marketing and retail strategy. In this project, I used a dataset containing advertising budgets and corresponding sales figures to predict future sales based on the TV advertisement budget.

The model was developed using **Simple Linear Regression** with Python and Scikit-learn.

---

## 📁 Dataset

- Dataset Name: **Advertising.csv**
- Features: `TV`, `Radio`, `Newspaper`
- Target: `Sales`
- Source: [Kaggle Dataset](https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input)

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

---

## 📌 Steps Involved

1. Data Loading and Cleaning
2. Exploratory Data Analysis (EDA)
3. Correlation Analysis
4. Building Simple Linear Regression model using `TV` as the predictor
5. Model Evaluation (R² Score, Mean Squared Error)
6. Visualization of regression line

---

## 📈 Model Results

- **R² Score:** ~0.81 (example, varies per run)
- **Mean Squared Error:** Low (indicating good fit)

---

## 🖥️ How to Run

1. Download this repository or clone it.
2. Place `Advertising.csv` and `sales_prediction.py` in the same folder.
3. Open terminal in that folder.
4. Run the following command:

```bash
python sales_prediction.py
