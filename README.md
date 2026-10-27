# PrivSynth-Rare: Differentially Private Synthetic Medical Data

**Live demo:** https://privsynth-rare-arpit.streamlit.app

A conditional VAE trained with DP-SGD (Opacus) generates synthetic patient records for a **rare disease class**. Each run is scored on three things:

- **Fidelity:** KS statistic and correlation difference between real and synthetic data.
- **Privacy:** membership-inference AUC, distance to the closest real record, and copy rate.
- **Usefulness:** a Random Forest trained on synthetic data only, and on real plus synthetic data, then tested on held-out real patients.

The data is the Wisconsin Diagnostic Breast Cancer dataset, with malignant cases cut down to 60 records (14.4%) so that malignant becomes the rare class.

## Run the web app locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Run the original experiment (no UI)

```bash
python main.py          # writes results.csv, synthetic CSVs and plots to ./out
```

Optional environment variables: `EPOCHS=200 N_SYN=2000 SEED=42`.

## App features

- **Pick a privacy level:** No privacy, ε = 10 / 5 / 1, or a custom ε from a slider.
- **Single run:** live privacy, fidelity and utility metrics, plus a training loss curve, a PCA comparison of real and synthetic data, ROC and precision–recall curves, and confusion matrices.
- **Full comparison:** results table, utility-vs-ε and attack-risk plots, loss curves and ROC curves for all four settings, with `results.csv` available to download.
- **Downloads:** synthetic data as a CSV file.

## Project structure

```
app.py            Streamlit UI
core.py           Pipeline (same logic as main.py, plus loss and ROC/PR outputs)
main.py           Original command-line experiment
notebooks/        Original Colab notebook
requirements.txt  CPU-only PyTorch + dependencies
```

## Limitations

- The DP guarantee covers only the training of the CVAE. The scaler statistics and clipping bounds are computed from the raw training data.
- Results come from a single seed, and the test set contains only 18 rare cases.
- This is an educational project and is not for clinical use.
