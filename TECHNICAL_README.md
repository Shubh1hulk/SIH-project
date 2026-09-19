# Quantum Disease Detector: Technical README

## System Purpose

This repository implements a hybrid quantum-classical heart-disease prediction
system. It combines a FastAPI inference service, a PyQt5/QML desktop client,
supervised preprocessing, three quantum model variants, tuned classical
baselines, and an out-of-fold stacked committee.

The default dataset is the Cleveland Heart Disease dataset. The target is
binarized as follows:

- `0`: no disease
- `1` through `4`: disease present

## Architecture

```text
CSV or QML patient form
          |
          v
FastAPI /predict endpoint
          |
          v
Imputer -> StandardScaler -> ANOVA top-4 selector -> [0, pi] angle scaling
          |
          +--> VQC: 4 qubits, 4 data re-uploading layers
          +--> Bagged VQC ensemble
          +--> Quantum kernel SVM
          +--> Tuned RBF SVM
          +--> Tuned logistic regression
                         |
                         v
              Logistic stacked meta-learner
                         |
                         v
              Probability, threshold, risk band
```

### Quantum models

- **VQC**: trainable rotation layers with ring entanglement and a multi-qubit
  Z-readout.
- **VQC ensemble**: bootstrap-trained VQC members whose probabilities are
  averaged to reduce variance.
- **Quantum kernel SVM**: fidelity kernel computed from the quantum feature
  map, followed by an SVM classifier with cross-validated `C`.

### Classical models

Random Forest, RBF SVM, and logistic regression are tuned with five-fold
stratified cross-validation on the training partition only.

### Stacking and leakage control

The meta-learner receives out-of-fold probabilities from all five members.
The operating threshold is selected from those out-of-fold predictions using
Youden's J statistic. The held-out test set is used only for final evaluation.

## Repository Layout

| Path | Responsibility |
| --- | --- |
| `backend/app.py` | FastAPI service, artifact loading, prediction endpoints |
| `backend/preprocess.py` | Imputation, scaling, feature selection, serialization |
| `backend/quantum_model.py` | VQC, ensemble, and quantum-kernel implementations |
| `backend/train.py` | End-to-end training, evaluation, artifact export, plots |
| `backend/artifacts/` | Serialized trained models and explainability data |
| `frontend/` | QML views and the API-aware desktop launcher |
| `data/heart_combined.csv` | Combined UCI dataset generated with provenance |
| `scripts/` | Accuracy, leakage, and dataset verification scripts |

## Installation

The tested dependency pins target Python 3.9 through 3.12. Python 3.14 can
require newer compatible package releases, especially for NumPy, SciPy,
scikit-learn, PennyLane, and PyQt5.

```bash
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

For an exact reproducible dependency graph on a supported Python version:

```bash
python -m pip install -r requirements-lock.txt
```

## Run the System

Train and export all artifacts:

```bash
python backend/train.py
```

On a headless Linux session, generate plots without opening a Qt window:

```bash
MPLBACKEND=Agg python backend/train.py
```

Start the API:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The health endpoint is available at `http://127.0.0.1:8000/health`.

Start the desktop client on Wayland:

```bash
QT_QPA_PLATFORM=wayland python frontend/run_ui.py
```

The launcher starts the API automatically when it is not already running.

## Current Cleveland Reference Metrics

The fixed 80/20 stratified split uses `random_state=42` and contains 242
training rows and 61 test rows.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Hybrid stacked committee | 0.8525 | 0.8519 | 0.8214 | 0.8364 |
| Tuned classical SVM | 0.8525 | 0.8519 | 0.8214 | 0.8364 |
| VQC ensemble | 0.8361 | 0.8214 | 0.8214 | 0.8214 |
| Random Forest | 0.8361 | 0.8214 | 0.8214 | 0.8214 |
| Logistic regression | 0.8361 | 0.8214 | 0.8214 | 0.8214 |
| Quantum kernel SVM | 0.7705 | 0.7500 | 0.7500 | 0.7500 |
| VQC | 0.7541 | 0.7407 | 0.7143 | 0.7273 |

The comparison plot is written to `plots/model_comparison.png`. The plot is a
regenerable output and is excluded from version control by default.

## API Contract

- `GET /health`: reports service health and whether artifacts loaded.
- `POST /predict`: scores one patient using the 13 clinical features.
- `POST /predict/csv`: scores a CSV upload with a header row.

The API returns a probability, binary prediction, risk level, recommendation,
and feature-importance information. This is a research prototype and is not a
medical diagnosis or a substitute for professional clinical judgment.

## Verification

```bash
python scripts/verify_combined.py
python scripts/verify_no_leakage.py
python scripts/verify_accuracy.py
```
