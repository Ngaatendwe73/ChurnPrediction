# Bank Customer Churn Prediction
### Zimbabwe Banking & MicroFinance — Industrial Attachment Project

An end-to-end machine learning project that predicts customer churn
for Zimbabwe's banking and microfinance sector. Built as part of an
industrial attachment, it covers the full pipeline from data generation
through feature engineering, model training, SHAP explainability, and
a live Flask REST API with a browser dashboard.

---

## Project Structure
---

## Zimbabwe-Specific Features

This project goes beyond a generic churn model by engineering
features that reflect Zimbabwe's unique banking context:

| Feature | Description |
|---|---|
| `RFM_Score` | Recency × Frequency × Monetary composite score |
| `Mobile_Money_Risk` | Likelihood customer has shifted to EcoCash/OneMoney (0–3) |
| `Digital_Adoption_Score` | Engagement with digital banking channels (0–4) |
| `Has_Zero_Balance` | Flag for zero-balance accounts (key churn signal) |
| `Account_Currency` | USD vs ZWL — inflation exposure |
| `Real_Balance_After_Inflation` | Balance adjusted for ~500% ZWL annual inflation |
| `Loyalty_Score` | Composite of tenure, products, and activity |

---

## Models Trained

| Model | AUC-ROC | F1 (churn) | Notes |
|---|---|---|---|
| Logistic Regression | 0.683 | 0.389 | Best on synthetic data |
| Random Forest | 0.672 | 0.367 | 300 trees, depth 8 |
| Gradient Boosting | 0.653 | 0.157 | 200 trees, lr 0.08 |
| XGBoost | 0.648 | 0.351 | scale_pos_weight balancing |

> **Note:** AUC values above are for the synthetic dataset.
> On the real Kaggle dataset, expect AUC of **0.85–0.87**.

All models use **SMOTE** for class imbalance and are evaluated with
**5-fold stratified cross-validation**. The best model is saved
automatically and served by the Flask API.

---

## How to Run

### 1. Clone the project

```bash
git clone https://github.com/Ngaatendwe73/ChurnPrediction.git
cd ChurnPrediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run each step in order

```bash
# Generate dataset
python step1_generate_data.py

# EDA + feature engineering
python step2_eda.py

# Train models + SHAP
python step3_model.py

# Start Flask API + dashboard
python step4_api.py
```

### 4. Open the dashboard
---

## Using the Real Kaggle Dataset

1. Go to: https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers
2. Sign in and download `Churn_Modelling.csv`
3. Place it in the `data/` folder
4. In `step2_eda.py`, `DATA_PATH` already points to `data/Churn_Modelling.csv` — no changes needed
5. Run from step 2 onwards

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Live dashboard |
| `GET` | `/health` | API status check |
| `POST` | `/predict` | Score one customer |
| `POST` | `/predict/batch` | Score many customers |

### Example — single prediction

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 620,
    "Age": 38,
    "Tenure": 3,
    "Balance": 45000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 35000,
    "Geography": "France",
    "Gender": "Male",
    "Account_Currency": "USD"
  }'
```

### Response

```json
{
  "churn_probability": 0.3352,
  "risk_band": "Medium",
  "action": "Schedule a retention check-in call.",
  "features_used": { ... }
}
```

---

## Risk Bands

| Band | Probability | Recommended Action |
|---|---|---|
| Low | < 30% | No immediate action |
| Medium | 30–50% | Schedule retention check-in |
| High | 50–70% | Priority outreach + incentive |
| Critical | > 70% | Immediate intervention |

---

## Tech Stack

- **Python 3.11+**
- **pandas / numpy** — data manipulation
- **scikit-learn** — modelling, preprocessing, evaluation
- **imbalanced-learn** — SMOTE oversampling
- **XGBoost** — gradient boosting
- **SHAP** — model explainability
- **Flask** — REST API and dashboard
- **matplotlib / seaborn** — visualisation
- **joblib** — model serialisation

---

## Author

**Ngaatendwe Masiku**  
Industrial Attachment Student — Data Science  
Zimbabwe

GitHub: [@Ngaatendwe73](https://github.com/Ngaatendwe73)