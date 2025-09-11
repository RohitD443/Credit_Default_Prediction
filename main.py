# Loan_Default_Prediction
"""
Loan Default Prediction System
--------------------------------
Author: Rohit Deshawal
Description: This script preprocesses credit scoring data and applies
             Logistic Regression, Decision Tree, and Random Forest
             models to predict loan default risk.
"""

# =============================
# 📌 Import Libraries
# =============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# =============================
# 📌 Load Dataset
# =============================
df = pd.read_csv("data/credit_score_data.csv")

# =============================
# 📌 Handle Missing Values
# =============================

# 1. Drop column with too many missing values
df = df.drop(columns=["Months since last delinquent"], errors="ignore")

# 2. Impute numerical columns with median
num_cols = ["Annual Income", "Credit Score", "Bankruptcies"]
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# 3. Impute categorical column with most frequent value
cat_cols = ["Years in current job"]
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# 4. Drop rows with missing target variable
df = df.dropna(subset=["Credit Default"])

# =============================
# 📌 Feature Engineering
# =============================
X = pd.get_dummies(df.drop("Credit Default", axis=1), drop_first=True)
y = df["Credit Default"]

# =============================
# 📌 Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=102, stratify=y
)

# =============================
# 📌 Feature Scaling
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# 📌 Model Training & Evaluation
# =============================

def evaluate_model(model, X_test, y_test, model_name):
    """Utility function to print evaluation metrics."""
    y_pred = model.predict(X_test)
    print(f"\n🔹 {model_name} Results")
    print("Accuracy Score :", round(accuracy_score(y_test, y_pred), 2))
    print("Precision Score:", precision_score(y_test, y_pred))
    print("Recall Score   :", recall_score(y_test, y_pred))
    print("F1 Score       :", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
evaluate_model(log_reg, X_test, y_test, "Logistic Regression")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=102)
dt.fit(X_train, y_train)
evaluate_model(dt, X_test, y_test, "Decision Tree")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
evaluate_model(rf, X_test, y_test, "Random Forest")
