# Bank Marketing — Prediction of Term Deposit Subscription

Binary classification on the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).  
**Goal**: predict whether a client will subscribe to a term deposit after a phone call campaign.

## Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| File | `data/bank-full.csv` |
| Rows | 45,211 |
| Features | 16 (7 numeric, 9 categorical) |
| Target | `y` — yes/no (binary) |
| Class balance | ~88% no / ~12% yes |

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline LogReg (7 num. features) | 0.888 | 0.569 | 0.161 | 0.251 | 0.831 | — |
| LogReg `class_weight='balanced'` | 0.845 | 0.416 | 0.812 | 0.550 | 0.908 | 0.537 |
| Random Forest | 0.840 | 0.411 | 0.852 | 0.554 | 0.921 | 0.594 |
| Gradient Boosting (base) | 0.907 | 0.654 | 0.438 | 0.524 | 0.927 | 0.606 |
| **Gradient Boosting (tuned)** | — | — | — | **0.555** | **0.932** | **0.625** |

> **Best model**: Gradient Boosting after `RandomizedSearchCV` (50 iterations, 5-fold stratified CV).  
> ROC-AUC and PR-AUC are the primary metrics due to class imbalance.

## Project Structure

```
├── data/
│   ├── bank-full.csv        # main dataset (45k rows)
│   └── bank-names.txt       # feature descriptions
├── eda_baseline.ipynb       # main notebook (5 sections)
└── requirements.txt
```

### Notebook sections

| Section | Content |
|---|---|
| 1 | EDA: distributions, correlations, class imbalance |
| 2 | Full preprocessing: binary/ordinal/OHE encoding, stratified split, scaling |
| 3 | Model comparison: LogReg vs Random Forest vs Gradient Boosting |
| 4 | Hyperparameter tuning: `RandomizedSearchCV` + `StratifiedKFold` |
| 5 | Advanced: optimal threshold, `sample_weight`, LightGBM, feature importance |

## How to Run

```bash
git clone https://github.com/ArkovOlen/uci_bank_marketing.git
cd uci_bank_marketing
pip install -r requirements.txt
jupyter notebook eda_baseline.ipynb
```

Run all cells top-to-bottom (`Cell → Run All`). Each section builds on the previous.

## Key Findings

- **Class imbalance (88/12)** makes Accuracy misleading — a model always predicting "no" gets 88%. Use F1, ROC-AUC, PR-AUC.
- **Feature `duration`** (call length) is the strongest predictor but is unknown before the call — excluded in a real deployment scenario.
- **Default threshold 0.5** is suboptimal for imbalanced data; the optimal threshold (found in Section 5) improves F1 significantly.
- **`class_weight='balanced'` / `sample_weight`** are cheap ways to improve Recall without adding data.
- **LightGBM** is a drop-in replacement for sklearn GBM: 5–10× faster, often more accurate.
