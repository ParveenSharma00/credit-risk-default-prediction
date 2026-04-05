# 🏦 Credit Risk Default Prediction

> Predicting whether a credit applicant will default within 12 months — built end-to-end from raw bureau data to a production inference script.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?style=flat-square&logo=scikit-learn)
![Notebooks](https://img.shields.io/badge/Notebooks-14-green?style=flat-square&logo=jupyter)
![AUC](https://img.shields.io/badge/Best%20AUC-0.8575-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)

---

## 📊 Results at a Glance

| Model | AUC | KS Statistic | Gini | CV AUC |
|:---|:---:|:---:|:---:|:---:|
| External Bureau Score *(baseline)* | 0.8152 | 0.486 | 0.630 | — |
| Gradient Boosting | 0.8533 | 0.584 | 0.707 | 0.847 |
| Random Forest | 0.8501 | 0.518 | 0.700 | 0.841 |
| **Logistic Regression** ✅ | **0.8575** | **0.521** | **0.715** | **0.854** |

> 🏆 Best model beats the pre-trained external bureau score by **+0.0423 AUC**

---

## 🎯 The Problem

Given credit bureau features at the time of application — predict which customers will stop repaying within the next 12 months.

| | |
|:---|:---|
| 📁 Training data | 28,397 accounts |
| 🧪 Test data | 4,000 accounts |
| 🔢 Raw features | 100 anonymised bureau features across 9 families |
| ⚠️ Challenge | 5.79% default rate — **16.3:1 class imbalance** |
| 📏 Metrics used | ROC-AUC, Average Precision, KS Statistic, Gini Coefficient |

> ❌ Accuracy is meaningless here — a naive all-zero model scores **94.2% accuracy** while catching zero defaults

---

## 🗂️ Project Structure
```
credit-risk-default-prediction/
│
├── 📓 Part1_Data_Reading_Validation.ipynb       # 10 integrity checks, NaN analysis
├── 📓 Part2_EDA.ipynb                           # Feature power ranking, signal analysis
├── 📓 Part3_Feature_Engineering.ipynb           # 100 → 198 features, missingness flags
├── 📓 Part4_1_Setup_and_Load_Data.ipynb         # Preprocessed arrays, saved artefacts
├── 📓 Part4_2_Evaluation_Framework.ipynb        # Shared metrics: AUC, KS, Gini, CV
├── 📓 Part4_3_Baseline_Bureau_Score.ipynb       # External benchmark deep-dive
├── 📓 Part4_4_Logistic_Regression.ipynb         # LR with coefficient analysis
├── 📓 Part4_5_Random_Forest.ipynb               # RF with OOB and importances
├── 📓 Part4_6_Gradient_Boosting.ipynb           # GBM with training loss curve
├── 📓 Part4_7_Model_Comparison_Selection.ipynb  # 3-tier selection framework
├── 📓 Part4_8_Best_Model_Deep_Dive.ipynb        # Threshold, calibration, capture curve
├── 📓 Part4_9_Artefacts_and_Inference.ipynb     # inference.py + dry-run validation
├── 📓 Part4_10_Final_Summary.ipynb              # Reflections and answers
│
├── 🐍 inference.py               ← score new applicants from CLI
├── 🤖 best_model.pkl             ← trained model (run notebooks to generate)
├── ⚙️  inference_config.pkl       ← preprocessing params + decision threshold
└── 📄 Credit_Risk_Report.docx    ← full write-up with all charts embedded
```

---

## ⚡ Quickstart

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy joblib matplotlib seaborn
```

### 2. Run notebooks in order
Run `Part1` → `Part4_10`. Each notebook saves outputs that the next one loads.

### 3. Score new applicants
```bash
python3 inference.py --input new_applicants.csv --output scores.csv
```

**Output columns:**

| Column | Description |
|:---|:---|
| `account_id` | Original identifier — passed through unchanged |
| `default_probability` | Predicted probability of default `(0.0 – 1.0)` |
| `high_risk_flag` | `1` if above decision threshold, else `0` |
| `risk_band` | `LOW` / `MEDIUM` / `HIGH` / `VERY HIGH` |

---

## 🔍 Part 1 — Data Validation

10 integrity checks before any modelling:

- ✅ Zero train/test leakage — no shared account IDs
- ✅ Zero duplicate account IDs across all files
- ✅ Consistent 5.79% default rate across both splits
- ✅ Bureau score range confirmed valid (437–1102, zero nulls)
- ⚠️ 2 unmatched rows in X_train — dropped harmlessly on merge

> 💡 **Key insight:** NaN in credit bureau data means the event *never occurred* — not that data is missing. A customer with `recency_1 = NaN` has never had a late payment event. That absence is itself a positive credit signal, not noise.

---

## 📈 Part 2 — Exploratory Data Analysis

Standalone AUC computed for **every numeric feature** across 9 families:

| Family | Features | Best AUC | What it captures |
|:---|:---:|:---:|:---|
| `score` | 1 | 0.815 | Compressed bureau signal — the benchmark |
| `utilization` | 18 | ~0.75 | % of credit limit used — strongest raw signal |
| `repayment_history` | 31 | ~0.73 | Past behaviour predicts future behaviour |
| `recency` | 12 | ~0.65 | Months since events — missingness also predictive |
| `balance` | 24 | ~0.60 | Outstanding amounts across product types |
| `spend_behaviour` | 5 | ~0.57 | Spending patterns |
| `credit_limit` | 3 | ~0.55 | Approved limits by product |

**Missingness confirmed as a signal in both directions:**
- `utilization` NaN → customer defaults *more* (thin-file applicant, no product)
- `recency` NaN → customer defaults *less* (event never occurred = clean record)

Both directions are real and both became explicit features in Part 3.

---

## ⚙️ Part 3 — Feature Engineering

All parameters fitted on **training data only** — zero leakage into test set.

| Step | Action | Before | After |
|:---:|:---|:---:|:---:|
| 1 | Binary missingness flags | 98 numeric cols | +79 binary flag cols |
| 2 | Median imputation | up to 94% NaN | 0 NaN remaining |
| 3 | One-hot encoding | 2 categorical cols | +21 dummy cols |
| 4 | Column alignment | variable schema | exact schema match |
```
Raw features:         100
After engineering:    198   ✓ zero NaN
```

**Class imbalance correction:**
`16.28×` sample weights applied to default rows during training — rebalances the loss function without touching or augmenting the data distribution.

---

## 🤖 Part 4 — Modelling

### Evaluation framework
All models evaluated on the **same** 5-fold stratified CV + identical test set.
External bureau score used as the benchmark every trained model must beat.

### Model designs

<details>
<summary><b>Logistic Regression</b> — winner ✅</summary>

- `C=0.05` — strong L2 regularisation for 198 noisy features
- `class_weight='balanced'` — handles 16.3:1 imbalance natively
- `StandardScaler` inside Pipeline — prevents data leakage during CV folds
- Signed coefficients give direct interpretability: which features push risk up or down

</details>

<details>
<summary><b>Random Forest</b></summary>

- 200 trees, `max_depth=10`, `min_samples_leaf=20`
- `class_weight='balanced_subsample'` — recomputes weights per bootstrap, better than global rebalancing for RF
- OOB score enabled as a free internal validation estimate

</details>

<details>
<summary><b>Gradient Boosting</b></summary>

- 100 trees, `learning_rate=0.08`, `max_depth=4`, `subsample=0.8`
- Sequential error correction — each tree fits residuals of all previous ones
- Sample weights passed directly to `.fit()` — GBM doesn't support `class_weight` natively

</details>

### Selection framework
```
Tier 1 — Must-have checks  :  beats bureau AUC  +  KS > 0.30  +  no overfitting
Tier 2 — Primary metric    :  highest test AUC
Tier 3 — Tiebreaker        :  highest CV AUC  (confirms generalisation)
```

---

## 🔬 Part 4.8 — Best Model Deep Dive

| Dimension | Finding |
|:---|:---|
| Threshold | Youden's J = 0.5463 — tunable based on business FN vs FP cost ratio |
| Calibration | Probabilities directionally accurate — suitable for risk-based pricing |
| Default capture | Top 20% riskiest applicants capture **~74% of all defaults** |
| Feature importance | Utilization + repayment history dominate; missingness flags carry real importance |
| Error analysis | Most false negatives scored just below threshold — borderline cases, not wide misses |

**Default capture vs bureau baseline:**

| Reject top % | Our model | Bureau baseline | Gain |
|:---:|:---:|:---:|:---:|
| 5% | ~38% | ~31% | +7 pp |
| 10% | ~56% | ~47% | +9 pp |
| 20% | ~74% | ~66% | +8 pp |
| 30% | ~86% | ~79% | +7 pp |

---

## 🚀 Inference Script

`inference.py` is fully production-ready:
```bash
python3 inference.py --input applicants.csv --output scores.csv
```

- ✅ Validates input schema before scoring — catches column mismatches early
- ✅ Applies full preprocessing pipeline (flags → imputation → OHE → alignment)
- ✅ Handles missing columns and unseen categories gracefully (adds zeros)
- ✅ Assigns risk bands automatically based on predicted probability
- ✅ Dry-run validated — output matches notebook predictions exactly

---

## 💡 Key Technical Decisions

**Why missingness flags instead of just imputing?**
Credit bureau NaN is domain-meaningful. Imputing it with a median loses the signal entirely. Binary flags preserve the information while median imputation fills the numeric value for downstream models. Both work together.

**Why sample weights instead of SMOTE?**
SMOTE generates synthetic minority samples that can introduce noise — especially problematic on anonymised features with intentional noise already added. Sample weights rebalance the loss function without altering or expanding the data distribution.

**Why Logistic Regression over Gradient Boosting?**
With strong L2 regularisation on 198 noisy features, LR's linear decision boundary generalised better to the test set. The interpretable signed coefficients are also a practical advantage — in credit risk, knowing which features push risk up or down matters operationally.

**Why AUC and KS over accuracy?**
A classifier predicting no default for everyone scores 94.2% accuracy while catching zero defaults. AUC, KS, and Gini are threshold-independent metrics that correctly evaluate models under class imbalance. KS is the credit industry standard alongside Gini.

---

## 🛠️ Requirements
```
scikit-learn >= 1.0
pandas       >= 1.3
numpy        >= 1.21
joblib       >= 1.0
matplotlib   >= 3.4
seaborn      >= 0.11
```

---

## 🏷️ Topics

`credit-risk` `machine-learning` `python` `scikit-learn` `feature-engineering` `imbalanced-classification` `logistic-regression` `random-forest` `gradient-boosting` `data-science` `fintech` `jupyter-notebook` `eda` `mlops`

---

<p align="center">
  Built with 🧠 by <b>Parveen Kumar Sharma</b>
</p>
