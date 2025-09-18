# =========================
# 1️⃣ Install & Import
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

# For pretty plots
sns.set(style="whitegrid", palette="Set2", font_scale=1.1)

# =========================
# 2️⃣ Load Data
# =========================
# ⬇️ Replace with your dataset path if in Google Drive
df = pd.read_csv("credit_score_data.csv")

# =========================
# 3️⃣ Handle Missing Values
# =========================
# Drop column with too many missing values (>50%)
df.drop(columns=["Months since last delinquent"], inplace=True)

# Impute numerical columns with median
num_cols = ["Annual Income", "Credit Score", "Bankruptcies"]
df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

# Impute categorical column with mode
cat_cols = ["Years in current job"]
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

# Drop rows with missing target
df.dropna(subset=["Credit Default"], inplace=True)

print("✅ Remaining Missing Values:\n", df.isnull().sum())

# =========================
# 4️⃣ Encode & Split
# =========================
X = pd.get_dummies(df.drop("Credit Default", axis=1), drop_first=True)
y = df["Credit Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =========================
# 5️⃣ Train Models
# =========================
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_prob_log = log_reg.predict_proba(X_test)[:,1]

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:,1]

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

# =========================
# 6️⃣ Evaluation Function
# =========================
def model_report(name, y_true, y_pred):
    print(f"\n📊 {name} Results")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 2))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

model_report("Logistic Regression", y_test, y_pred_log)
model_report("Decision Tree", y_test, y_pred_dt)
model_report("Random Forest", y_test, y_pred_rf)

# =========================
# 7️⃣ Visualizations
# =========================

# --- Target distribution
plt.figure(figsize=(5,4))
sns.countplot(x=y, palette="Set2")
plt.title("Distribution of Target: Credit Default")
plt.xlabel("Credit Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()
print("\n")
print("\n")

# --- Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()
print("\n")
print("\n")

# --- Model performance comparison
metrics = ['Accuracy','Precision','Recall','F1']
log_scores = [
    accuracy_score(y_test,y_pred_log),
    precision_score(y_test,y_pred_log),
    recall_score(y_test,y_pred_log),
    f1_score(y_test,y_pred_log)
]
dt_scores = [
    accuracy_score(y_test,y_pred_dt),
    precision_score(y_test,y_pred_dt),
    recall_score(y_test,y_pred_dt),
    f1_score(y_test,y_pred_dt)
]
rf_scores = [
    accuracy_score(y_test,y_pred_rf),
    precision_score(y_test,y_pred_rf),
    recall_score(y_test,y_pred_rf),
    f1_score(y_test,y_pred_rf)
]

score_df = pd.DataFrame({
    'Logistic Regression': log_scores,
    'Decision Tree': dt_scores,
    'Random Forest': rf_scores
}, index=metrics)

score_df.plot(kind='bar', figsize=(8,5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.show()
print("\n")
print("\n")

# --- ROC Curves
plt.figure(figsize=(6,5))
for name, y_prob in {
    'Logistic Regression': y_prob_log,
    'Decision Tree': y_prob_dt,
    'Random Forest': y_prob_rf
}.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

