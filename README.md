<div align="center">

# 📡 Churn Radar

### Predict which telecom customers are likely to leave, and what to offer them before they do.

![Python](https://img.shields.io/badge/Python-3-blue)
![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)

[**Live demo**](#) · [**Demo video**](#) · [**Notebook**](data.ipynb)

<img src="images/predict_high.png" width="85%" alt="Churn Radar prediction page showing a high-risk customer">

</div>

---

## The problem

Winning a new customer costs far more than keeping an existing one. If a telecom company can spot customers who are about to leave, it can send a targeted retention offer in time. This project builds a machine learning model that estimates each customer's **probability of churn**, labels them **Low, Medium or High risk**, and suggests the next step for the retention team.

## What the app does

| Predict churn | Insights and model |
|---|---|
| Enter a few key details (tenure, contract, internet service, payment method, monthly bill) | KPIs, churn charts, feature importance and model comparison |
| Get a churn probability on a gauge, a risk level and suggested actions | Shows which factors drive churn in the data |

<p align="center">
  <img src="images/predict_low.png" width="45%" alt="Low risk prediction">
  <img src="images/insights.png" width="45%" alt="Insights dashboard">
</p>

Extra customer details (gender, partner, add-on services and so on) sit in an optional section, so the model always receives every feature it was trained on.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data on Kaggle): **7,043 customers**, 21 columns, with an overall churn rate of **26.5%**.

## Approach

1. **Cleaning.** Converted `TotalCharges` from text to numbers (blank values belong to brand-new customers), encoded the target, and merged "No internet service" and "No phone service" into "No".
2. **Feature engineering.** Tenure groups, number of add-on services, average monthly spend, and the gap between current and average monthly charge.
3. **Modeling.** Stratified 80/20 train-test split with a preprocessing pipeline. Compared Logistic Regression, Random Forest and XGBoost, handling class imbalance with class weights.
4. **Evaluation.** Precision, recall, F1 and ROC-AUC, with the focus on **recall**: missing a customer who leaves costs far more than sending one extra retention offer.
5. **Explainability.** Feature importance in the app and SHAP analysis in the notebook.
6. **Deployment.** A Streamlit app with a prediction page and an insights page.

## Results

Held-out test set (20% of customers):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.498 | 0.791 | 0.612 | 0.842 |
| Random Forest | 0.564 | 0.650 | 0.604 | 0.827 |
| XGBoost | 0.516 | 0.781 | 0.621 | 0.837 |

## Key findings

- **Contract type matters most.** Month-to-month customers churn at **42.7%**, compared with 11.3% on one-year and **2.8%** on two-year contracts.
- **New customers are the most likely to leave.** Churn is **47.4%** in the first 12 months and falls to **9.5%** after four years.
- **Fiber optic customers churn the most** (41.9%), while customers with no internet service churn at 7.4%.
- The model's top features are a month-to-month contract, a two-year contract, fiber optic internet and short tenure.

## Risk levels

| Churn probability | Risk level |
|---|---|
| Below 35% | Low |
| 35% to 60% | Medium |
| 60% and above | High |

## Run it locally

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
├── app.py              # Streamlit app (prediction + insights)
├── data.ipynb          # EDA, feature engineering, modeling, evaluation, SHAP
├── churn_model.pkl     # Trained pipeline (preprocessing + model)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── images/             # Screenshots used in this README
├── requirements.txt
└── README.md
```

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Plotly, Streamlit, joblib.

## Author

**[Your Name]**
[LinkedIn](#) · [GitHub](#) · [Email](mailto:your.email@example.com)
