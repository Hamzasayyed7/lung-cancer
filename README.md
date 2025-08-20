# Lung Cancer Classification — Decision Tree + PCA (≥95% variance)

## How to run
```bash
pip install -U scikit-learn pandas numpy
python lung_cancer_dt_pca.py --data lung_cancer_examples.csv  # add --target NAME if needed
```

Artifacts are saved into `outputs/`:
- `baseline_report.txt` and `pca_report.txt`: per-class precision/recall/F1
- `metrics_compare.csv`: side-by-side metrics
- `confusion_matrix_*.csv`: confusion matrices
- `feature_importance_baseline.csv`: top features used by the baseline tree

## Notes
- The script automatically guesses the label column. If it’s wrong, pass `--target`.
- PCA is applied after making the data dense; it keeps components that together explain ≥95% variance.
- For best marks you can mention limitations and next steps (grid-search tree depth, cross-validation, handling class imbalance).
