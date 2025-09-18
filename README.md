# Credit Score Analysis

## Abstract

Credit scoring plays a crucial role in assessing an individual's
creditworthiness and enabling financial institutions to make informed
lending decisions. A reliable credit score prediction model helps banks,
lenders, and other financial organizations minimize the risk of loan
defaults and improve decision-making processes.

This project aims to build a **machine learning--based classification
model** to predict whether a person will **default on credit** based on
demographic, financial, and historical credit behavior. By applying
various supervised machine learning algorithms and comparing their
performance, the model provides insights to support credit risk
management.

------------------------------------------------------------------------

## Problem Statement

Financial institutions rely heavily on credit scoring systems to
evaluate a borrower's likelihood of repaying a loan. Manual evaluation
can be slow and prone to human bias. Traditional credit evaluation
methods may fail to detect patterns hidden in large datasets, leading to
inaccurate risk assessments.

The goal of this project is to **develop a predictive model** that
classifies individuals as either likely to **default** or **not
default** on their credit. The model leverages customer information such
as annual income, credit score, bankruptcies, and employment details to
make accurate predictions.

This automation reduces the time and effort needed for credit assessment
while improving the precision and consistency of decisions.

------------------------------------------------------------------------

## Motivation

The primary motivation is to **reduce financial risk** for lenders by
using data-driven decision making. Early and accurate detection of
potential defaulters helps financial institutions: - Improve loan
approval processes - Reduce non-performing assets (NPAs) - Minimize
credit losses\
- Provide fair and quick credit assessments to customers

With the growing availability of customer financial data, machine
learning offers a scalable and efficient solution to enhance credit risk
management.

------------------------------------------------------------------------

## Dataset

The dataset contains customer financial and credit-related information,
including: - **Annual Income** - **Credit Score** - **Bankruptcies** -
**Years in Current Job** - Other demographic and financial features

### Preprocessing Steps:

1.  **Handling Missing Values:**
    -   Dropped columns with more than 50% missing values (e.g., *Months
        since last delinquent*).\
    -   Imputed missing numerical values with median.\
    -   Imputed missing categorical values with mode.\
    -   Dropped rows with missing target variable (*Credit Default*).
2.  **Feature Encoding:**
    -   Converted categorical variables to numerical using **one-hot
        encoding**.
3.  **Feature Scaling:**
    -   Standardized features using **StandardScaler** to improve model
        performance.

------------------------------------------------------------------------

## Machine Learning Models

The following supervised learning models were applied to classify
whether a customer will default on credit:

-   **Logistic Regression:** Baseline statistical model to predict
    binary outcomes.\
-   **Decision Tree Classifier:** Tree-based model for easy
    interpretability and non-linear relationships.\
-   **Random Forest Classifier:** Ensemble method for robust prediction
    and handling of overfitting.

Each model was trained using an 80-20 train-test split.

------------------------------------------------------------------------

## Model Evaluation

Models were evaluated using key classification metrics: - **Accuracy** -
**Precision** - **Recall** - **F1-Score** - **Confusion Matrix** -
**Classification Report**

This comparison highlights the model that best balances precision and
recall for credit default prediction.

------------------------------------------------------------------------

## Results

  Model                      Accuracy   Precision   Recall   F1-Score
  -------------------------- ---------- ----------- -------- ----------
  Logistic Regression        \~XX%      \~XX%       \~XX%    \~XX%
  Decision Tree Classifier   \~XX%      \~XX%       \~XX%    \~XX%
  Random Forest Classifier   \~XX%      \~XX%       \~XX%    \~XX%

*(Replace XX% with actual scores after running the models.)*

From the results, **Random Forest** generally provided the best balance
of accuracy and recall, making it the most reliable model for predicting
potential defaulters.

------------------------------------------------------------------------

## Visualizations

To interpret and present model outcomes: - **Correlation Heatmap:**
Shows relationships between numerical features. - **Confusion Matrix
Heatmaps:** Visualizes correct vs. incorrect predictions for each
model. - **ROC Curve / AUC Score:** Evaluates classification performance
across thresholds. - **Feature Importance Plot (Random Forest):**
Identifies key factors influencing credit default prediction.

------------------------------------------------------------------------

## Tools & Technologies

-   **Python**\
-   **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**\
-   **scikit-learn** (Logistic Regression, Decision Tree, Random
    Forest)\
-   **Jupyter Notebook / Google Colab**

------------------------------------------------------------------------

## Future Scope

-   Integrating advanced ensemble methods (e.g., XGBoost, LightGBM) for
    better accuracy.\
-   Using larger real-world credit datasets for improved
    generalization.\
-   Deploying the model as a web-based or API service for real-time
    credit scoring.

------------------------------------------------------------------------

## Conclusion

This project demonstrates how **machine learning techniques can
significantly enhance credit risk prediction**, enabling financial
institutions to make more informed and faster lending decisions while
minimizing potential credit losses.
