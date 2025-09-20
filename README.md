📊 Credit Score Analysis
✨ Abstract

Credit scoring is essential for evaluating an individual's creditworthiness and enabling banks & financial institutions to make informed lending decisions.
This project builds a machine learning–based classification model to predict whether a person will default on credit, using demographic, financial and historical credit data.
By applying and comparing multiple supervised ML models, the project delivers insights to support better risk management.

🎯 Problem Statement

Traditional credit evaluation methods can be slow and inconsistent, sometimes missing subtle patterns in large datasets.
The objective is to develop a predictive model that classifies individuals as likely to default or not default on their credit.
This automation helps:

⏱ Speed up loan approvals

🎯 Improve decision accuracy

💰 Reduce financial losses from defaults

💡 Motivation

🔍 Early and accurate detection of potential defaulters reduces risk for lenders.

📈 Data-driven decisions improve consistency and fairness.

🤖 Machine learning offers scalable, automated credit risk assessment.

🗂️ Dataset

The dataset contains key customer details such as:

Annual Income

Credit Score

Bankruptcies

Years in Current Job

Other demographic & financial features

🔧 Preprocessing Steps

Missing Values:
➤ Dropped columns with >50% missing values (e.g., Months since last delinquent).
➤ Imputed numerical columns with median and categorical with mode.
➤ Dropped rows with missing target variable (Credit Default).

Feature Encoding: One-hot encoding for categorical variables.

Scaling: StandardScaler for normalized numerical features.

🧠 Machine Learning Models

Three supervised learning models were trained and compared:

Model	Description
⚡ Logistic Regression	Baseline statistical model for binary classification
🌳 Decision Tree	Interpretable tree-based model
🌲 Random Forest	Ensemble of decision trees for robust prediction

Each model used an 80-20 train-test split.

📈 Model Evaluation

Models were evaluated with:

✅ Accuracy

✅ Precision

✅ Recall

✅ F1-Score

✅ Confusion Matrix

✅ ROC–AUC Curve

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	~XX%	~XX%	~XX%	~XX%
Decision Tree	~XX%	~XX%	~XX%	~XX%
Random Forest	~XX%	~XX%	~XX%	~XX%

(Replace XX% with your final scores.)

📊 Visual Insights
🔥 Correlation Heatmap

<img width="995" height="744" alt="Corelation Heat Map" src="https://github.com/user-attachments/assets/ab2f83a0-0947-471b-a64b-892379ac9867" />

✅ Confusion Matrix (per Model)

Logistic Regression ||	Decision Tree || Random Forest

<img width="703" height="458" alt="Model Performance Comparision" src="https://github.com/user-attachments/assets/677e3782-b934-4c58-92dc-1760afccb699" />
	
📉 ROC–AUC Curve

<img width="548" height="482" alt="ROC Curve" src="https://github.com/user-attachments/assets/9f0d0778-39c2-4265-9340-ff1d370c8fac" />

Compares model performance across classification thresholds.

🌟 Feature Importance (Random Forest)

Highlights the most influential factors in predicting credit default.

🛠 Tools & Technologies

Python 🐍

Pandas, NumPy, Matplotlib, Seaborn 📊

scikit-learn ⚙️

Jupyter Notebook / Google Colab 💻


✅ Conclusion

Machine learning enables fast, accurate credit risk predictions, helping financial institutions:

Make smarter lending decisions

Reduce non-performing assets (NPAs)

Enhance overall financial stability 💹.
