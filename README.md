# PFR Conversion & Liquid-Composition Predictor

A Streamlit app that predicts, **from inlet conditions only**, the outcome of the
plug-flow reactor (PFR) run:

- **Liquid composition** (RM4 / RM2 / RM5 / RM3, w/w %) — the reliable output
- **Expected conversion** — shipped, but flagged as low-confidence (see below)

Trained on 35 plant runs (`Model_1_0_Data.xlsx`).

## What the model does and doesn't do

| Target | LOOCV R² | Verdict |
|---|---|---|
| Conversion | ≈ 0.01 | **Not predictable** from inlet conditions — barely beats guessing the plant average. Shown with a clear warning. |
| RM4 % | 0.58 | Reliable |
| RM2 % | 0.53 | Reliable |
| RM5 % | 0.60 | Reliable |
| RM3 % | 0.49 | Reliable |

Composition is genuinely learnable; conversion is a known **data-intrinsic**
limitation, not a modelling bug. The app makes this explicit instead of hiding it.

## Design decisions

- **Forward / inlet-only prediction.** Outlet temperatures, outlet pressure,
  pressure drop, and post-reaction liquid/vapour quantities are **excluded** —
  they reflect the reaction outcome and would leak the answer (conversion is in
  fact derived from the liquid/vapour split).
- **Excluded per spec:** average RM1+RM2 rate, average RM2 rate, average RM3 rate.
- **Constant columns dropped automatically:** avg RM3 density, pipe diameter, volume.
- **13 features available, reduced to 4.** Feature selection showed that
  composition accuracy (mean LOOCV R² ≈ 0.55) is almost entirely carried by
  four inlet variables; the other nine add negligible signal (collinear or weak).
  **Inputs used:** PFR inlet temperature, PFR inlet pressure, RM3 molar ratio
  wrt RM1, residence time.
- **Models:** Ridge regression on standardised features. α=5 for composition
  (multi-output), α=30 for conversion (heavily regularised so it behaves honestly
  near the mean). Evaluated by **leave-one-out CV** — the only honest scheme at n=35.
- Composition predictions are clipped at 0 and renormalised to sum to 100 %.

## Files

```
app.py                 Streamlit front-end
train.py               Reproducible training -> model_bundle.joblib
data_utils.py          Shared data loading + feature contract
model_bundle.joblib    Pre-trained models + metadata (app runs out of the box)
Model_1_0_Data.xlsx    Source data
requirements.txt       Pinned dependencies
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain after updating the data:

```bash
python train.py        # regenerates model_bundle.joblib
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repo (keep all files, including
   `model_bundle.joblib` and `Model_1_0_Data.xlsx`).
2. Go to https://share.streamlit.io → **New app**.
3. Point it at your repo, branch, and `app.py` as the entry file.
4. Deploy. `requirements.txt` is picked up automatically.

> The pinned versions (scikit-learn 1.8.0, numpy 1.26.4, pandas 2.2.3) match the
> environment the model was pickled in, so the bundle loads without warnings.
> The app does **not** need the xlsx at runtime — it loads `model_bundle.joblib`.
