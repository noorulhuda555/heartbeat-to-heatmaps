# Heartbeat to Heatmaps

This repository contains a full data-mining assignment with:

- Heart Disease tabular analytics (Parts A, B, C, E)
- MNIST CNN pipeline (Part D)
- A local Streamlit dashboard for interactive heart-risk prediction

## Project Structure

- `task.ipynb` - Main notebook containing Parts A-D
- `ui.py` - Streamlit front-end dashboard (Part E)
- `heart_disease_uci.csv` - Heart disease dataset
- `requirements.txt` - Python dependencies
- `report.pdf` - Short module-wise report

## Environment Setup (Reproducibility)

Use Python 3.10+ (recommended).

### 1) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## Run Notebook (Parts A-D)

```powershell
jupyter notebook task.ipynb
```

Then run cells in order from top to bottom.

### Recommended execution order inside notebook

1. Preprocessing section
2. Part A
3. Part B
4. Part C
5. Part D

If you see stale variable errors, use **Kernel -> Restart & Run All**.

## Run Front-End Dashboard (Part E)

From this folder:

```powershell
streamlit run ui.py
```

Open the localhost URL shown in terminal (usually `http://localhost:8501`).

## Notes for Graders

- The dashboard includes 13 labeled inputs and a prefilled real test patient.
- Click **Predict** directly to test output.
- Output includes:
  - predicted class + risk color
  - confidence score
  - top-3 feature drivers chart
  - nurse-friendly explanation

## Determinism / Randomness

Where possible, `random_state=42` or framework seeds are used.
Minor variation can still occur across hardware/library versions.

## Common Troubleshooting

1. **Module not found**Re-check active environment and run:

   ```powershell
   pip install -r requirements.txt
   ```
2. **TensorFlow / SHAP version behavior differences**Re-run cells from the relevant section after kernel restart.
3. **Streamlit import warning in IDE**
   Usually a linter environment issue. Runtime works after dependency install.

## Single-command run for UI

For Part E grading, use:

```powershell
streamlit run ui.py
```
