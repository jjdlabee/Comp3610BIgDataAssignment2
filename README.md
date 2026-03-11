# NYC Taxi Tip Prediction
**COMP 3610 — Big Data | Assignment #2**

Predicts NYC yellow taxi tip amounts and classifies high-tip trips using the January 2024 TLC trip dataset. Covers the full ML pipeline: data ingestion, feature engineering, model training, hyperparameter tuning, neural network implementation, and evaluation.

---

## Project Structure

```
Comp3610BigDataAssignment2/
├── notebook.ipynb             ← Main analysis notebook
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── .gitignore                 ← Excludes data, models, venv
├── data/
│   └── raw/                   ← Downloaded parquet + CSV (git-ignored)
└── models/                    ← Saved model artifacts (git-ignored)
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Comp3610BigDataAssignment2
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create required directories
```bash
mkdir -p data/raw models
```

### 5. Run the notebook
```bash
jupyter notebook notebook.ipynb
```

> **Note:** The notebook downloads data automatically on first run (~700 MB). Ensure you have a stable internet connection and at least 8 GB of free RAM before running all cells.

---

## Data Sources

| File | Source | Size |
|---|---|---|
| `yellow_tripdata_2024-01.parquet` | [TLC Trip Data](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet) | ~700 MB |
| `taxi_zone_lookup.csv` | [TLC Zone Lookup](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv) | ~12 KB |

Data is downloaded programmatically in the notebook and saved to `data/raw/`. These files are excluded from version control via `.gitignore`.

---

## Tasks Covered

| # | Task | Description |
|---|---|---|
| 1 | Data Ingestion | Programmatic download, lazy loading with Polars |
| 2 | Preprocessing | Null handling, borough joins, feature engineering |
| 3 | Target Variables | `tip_amount` (regression), `high_tip` (classification) |
| 4 | Data Splitting | Stratified 70/15/15 split, StandardScaler |
| 5 | Regression Models | Linear Regression, Random Forest Regressor |
| 6 | Classification Models | Logistic Regression, Random Forest Classifier |
| 7 | Hyperparameter Tuning | RandomizedSearchCV on best model |
| 8 | Cross-Validation | 5-fold stratified CV on 300k sample |
| 9 | Neural Network | PyTorch feedforward network, 20 epochs |
| 10 | Evaluation | ROC curves, confusion matrix, residual analysis |
| 11 | Feature Importance | Random Forest importances, regression coefficients |
| 12 | Written Analysis | Model comparison, limitations, improvements |

---

## Requirements

- Python 3.10+
- ~8 GB RAM recommended
- GPU optional (CUDA-compatible for faster neural network training)

---

## Key Results

| Task | Best Model | Key Metric |
|---|---|---|
| Regression | Random Forest Regressor | R² = 0.689, MAE = $1.02 |
| Classification | Random Forest Classifier (tuned) | F1 = 0.8627, AUC = 0.8024 |
| Neural Network | TipClassifier (PyTorch) | F1 = 0.8637, AUC = 0.8000 |