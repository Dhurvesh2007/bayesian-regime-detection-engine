# Bayesian Regime Detection Engine for Equity Direction Forecasting

Professional GitHub-ready implementation aligned to the supplied Zetheta project brief.

## What this repository contains
- `run_model.py` — end-to-end model execution
- `dashboard.py` — professional Streamlit regime intelligence dashboard
- `src/` — HMM, Bayesian transition and post-processing modules
- `data/` — supplied demonstration dataset and processed features
- `models/` — fitted model artifact
- `outputs/` — predictions, diagnostics and figures
- `notebooks/` — full model notebook
- `docs/` — data provenance and implementation notes

## Run locally
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_model.py
streamlit run dashboard.py
```

Open the Streamlit URL shown in the terminal.

## Five regimes
Risk-On · Risk-Off · Transitional · Late-Cycle · Post-Shock

## Data integrity
The supplied archive is explicitly labelled **SYNTHETIC DEMONSTRATION DATA — NOT REAL MARKET DATA**. It is included to make the pipeline reproducible. Do not use its numerical outputs as real-market evidence or investment advice.

## Important scope note
The repository is the runnable core and dashboard. The original brief additionally requests advanced Bayesian MCMC, full RS-VAR, Bayesian deep learning, two foundation models, sequential Monte Carlo/BOCPD, conformal variants, BMA/stacking, TDA/GNN, Monte-Carlo risk and extensive validation. Those modules should be added and validated before claiming full production completion.
