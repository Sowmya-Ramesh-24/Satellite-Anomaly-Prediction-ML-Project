# Satellite Anomaly Prediction

Predict **whether a satellite's next anomaly happens within 7 days** (`HORIZON_DAYS` in `src/config.py`)
from space weather, orbit and anomaly history. Inspired by the CS229 project *"Satellite Anomaly Prediction
using Survival Analysis and Machine Learning"* (Naughton, 2019), using the same three model families
(linear model, Naive Bayes, SVM) with satellite-grouped validation and leak-free features.

## Repo layout
```
data/raw/            anom5j.xls (NOAA anomalies), satcat.csv (Celestrak), sunspot.csv (daily sunspot number)
notebooks/
  exploration.ipynb         look at the raw data
  lr.ipynb                  Logistic Regression
  naive_bayes.ipynb         Gaussian Naive Bayes
  svm.ipynb                 RBF SVM
  results_comparison.ipynb  side-by-side comparison of all models
src/
  config.py                 paths, horizon, features, CV settings
  preprocessing/            load_data.py, clean_data.py, create_tte.py (gap table + label)
  evaluation/               cv.py (grouped CV), metrics.py, plots.py
results/                    predictions_*.csv, metrics_comparison.csv, ROC / confusion / metric charts
run.py                      executes all notebooks in order
```
Each model notebook follows: 1 Imports -> 2 Datasets -> 3 Preprocessing -> 4 Define model -> 5 Train -> 6 Plots and results
(accuracy, precision, recall, F1, ROC AUC plot, confusion matrix). Each saves its out-of-fold predictions to
`results/`, and `results_comparison.ipynb` loads them so every model is scored on identical samples and folds.

## Run
```bash
pip install -r requirements.txt
python run.py          # or open the notebooks and run them top to bottom (models first, comparison last)
```

## Method
- **Samples:** one row per gap between consecutive anomalies of a satellite (58 satellites with >= 10 anomaly-days, 2,871 samples; same-day reports collapsed; pre-1976 anomalies and Space Shuttle missions removed).
- **Label:** 1 if the next anomaly occurs within 7 days (48.9% of samples, so classes are balanced).
- **Features** (all known at the start of the gap): month, sunspot mean over prior 27 / 81 days and their difference, orbit class and altitude, previous gap length, number of previous anomalies.
- **Validation:** 5-fold **GroupKFold by satellite** (a satellite is never in both train and test). Precision/recall/F1/confusion matrix use a 0.5 threshold.
- `ACOMMENT` is not used: it records space-weather conditions at the time of the anomaly (leakage).

## Results
| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.603 | 0.595 | 0.586 | 0.591 | 0.641 |
| Naive Bayes | 0.555 | 0.538 | 0.636 | 0.583 | 0.592 |
| SVM (RBF) | 0.598 | 0.600 | 0.536 | 0.566 | 0.638 |
| Majority-class baseline | 0.511 | 0.000 | 0.000 | 0.000 | 0.500 |

Per-fold ROC AUC (mean +- std): LR 0.646 +- 0.028, NB 0.616 +- 0.058, SVM 0.646 +- 0.049.

- All models beat the baseline, but only modestly (AUC ~ 0.64). Logistic Regression and SVM are indistinguishable given the fold-to-fold spread; Naive Bayes is clearly weaker.
- Earlier experiments (time-to-event regression, see git history / notes) showed that sunspot number, season and orbit class carry almost no signal; most of the predictive power comes from **anomaly history** (anomalies cluster in bursts).

## Limitations
- Space weather is the daily sunspot number only (no X-ray / particle flux); the database ends in 1994.
- satcat matches only ~10 of the 58 satellites (many are anonymised) and has no mass, so satcat features are not used in the model notebooks.
- Censoring is rare in this data (0.3% of gaps) because the database does not record when observation of a satellite ended.
- A 7-day horizon is a design choice; change `HORIZON_DAYS` and re-run to test others.
