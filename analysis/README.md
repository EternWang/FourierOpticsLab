# Analysis

This folder contains the reproducible Python workflow behind the numbers in [report/main.pdf](../report/main.pdf).

## Review options

### 1. Executed notebook
Open [FourierOptics_Analysis.ipynb](./FourierOptics_Analysis.ipynb).

The notebook is already executed, so GitHub renders the figures and printed values directly in the browser.

### 2. Standalone script

```bash
python -m venv .venv
pip install -r analysis/requirements.txt
python analysis/analyze.py
```

Generated outputs:
- `data/processed/results.json`: full structured summary
- `data/processed/grating_results.csv`: compact method-comparison table
- `analysis/output/screen_fit.png`: through-origin regression figure
- `analysis/output/screen_residuals.png`: residual diagnostics
- `analysis/output/random_uncertainty_budget.png`: random-uncertainty comparison
- `analysis/output/grating_method_comparison.png`: period estimates with propagated uncertainties

## What this demonstrates

- Constrained regression for the screen-angle method
- Analytic uncertainty propagation across a calibration chain
- Comparison of independent measurement methods
- Clean export of results to CSV and JSON for reuse
