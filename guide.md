# EEG AD/HC Phase 1 Guide

## 1. Project Goal

This repository implements a Phase 1 research prototype for EEG Signal Representation Learning using OpenNeuro `ds004504`.

It is limited to:

- OpenNeuro `ds004504`
- Resting-state closed-eyes EEG
- AD vs Healthy Control binary classification
- FTD exclusion
- EEG preprocessing
- EEGNet baseline training
- Subject-level evaluation
- 256D L2-normalized `z_eeg`
- Gradio research demo

Use safe wording: the model predicts `Alzheimer's EEG Pattern`, not a clinical diagnosis.

This guide defaults to local execution from the project folder. For Modal Notebook upload, persistent storage, and `L40S` resource setup, see [setup.md](setup.md).

## 2. Local Machine Quick Start

Run these commands from the project folder:

```text
C:\Users\skspa\Downloads\cookie\eeg_ad_hc_phase1_fixed\eeg_ad_hc_phase1_fixed
```

Windows PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Local output folders:

```text
data/raw/openneuro_ds004504/          raw OpenNeuro dataset
data/labels/                         generated labels
data/splits/                         subject-level folds
data/processed/epochs/               preprocessed epoch tensors
data/processed/preprocessing_logs/    preprocessing logs
models/checkpoints/                  trained checkpoints
models/embeddings/                   z_eeg embeddings
outputs/reports/                     metrics and JSON reports
outputs/figures/                     plots
outputs/gradio_cache/                Gradio-display copies
```

Full local run order:

```powershell
python -m src.data.prepare_labels --config configs/preprocessing.yaml
python -m src.data.create_subject_splits --config configs/training.yaml
python -m src.evaluation.leakage_check --config configs/training.yaml
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all --force
python -m src.evaluation.validate_preprocessed_shapes --config configs/training.yaml
python -m src.training.train_eegnet --config configs/training.yaml --fold 0
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --fold 0 --no-auto-train
python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-001 --no-auto-train
python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-001 --fold 0
python -m app.gradio_app
```

Or run the fold 0 pipeline with the included PowerShell helper:

```powershell
.\run_local_fold0.ps1 -SubjectId sub-001
```

To reuse existing preprocessing or training outputs:

```powershell
.\run_local_fold0.ps1 -SubjectId sub-001 -SkipPreprocess
.\run_local_fold0.ps1 -SubjectId sub-001 -SkipPreprocess -SkipTrain
```

Open the local Gradio URL printed by the last command, usually:

```text
http://127.0.0.1:7860
```

For a public temporary Gradio link, use:

```powershell
$env:GRADIO_SHARE="1"
python -m app.gradio_app
```

## 3. Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Modal or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Download Dataset

Recommended OpenNeuro command:

```bash
pip install -q --upgrade openneuro-py
openneuro-py download --dataset ds004504 --target-dir data/raw/openneuro_ds004504
```

Alternative GitHub source:

```bash
git clone https://github.com/OpenNeuroDatasets/ds004504.git data/raw/openneuro_ds004504
```

Expected Phase 1 groups:

- Healthy Control / `C`: 29 included, label `0`
- Alzheimer's Disease / `A`: 36 included, label `1`
- Frontotemporal Dementia / `F`: 23 excluded, label `-1`

The dataset size is suitable for Phase 1 proof-of-concept and baseline EEG representation learning. However, further validation with larger and external datasets is required before clinical use.

## 5. Prepare Labels

```bash
python -m src.data.prepare_labels --config configs/preprocessing.yaml
```

Outputs:

- `data/labels/participants_processed.csv`
- `data/labels/ad_hc_subjects.csv`
- `data/labels/excluded_ftd_subjects.csv`
- `data/labels/label_mapping.json`

## 6. Create Leakage-Safe Subject Splits

```bash
python -m src.data.create_subject_splits --config configs/training.yaml
python -m src.evaluation.leakage_check --config configs/training.yaml
```

Outputs:

- `data/splits/subject_level_5fold.csv`
- `data/splits/split_summary.json`
- `outputs/reports/leakage_check.json`

Important rule: epochs from the same subject must never appear in both training and validation/test sets.

## 7. Preprocess EEG

Preprocess all labeled AD/HC subjects:

```bash
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all
```

Force-refresh stale preprocessing outputs after changing channels, epoch length, or filtering:

```bash
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --all --force
```

Preprocess one subject:

```bash
python -m src.preprocessing.preprocess_subject --config configs/training.yaml --subject_id sub-001
```

Pipeline:

1. Channel selection and montage setup
2. Band-pass filtering `0.5-40 Hz`
3. Artifact/noise rejection using peak-to-peak amplitude threshold
4. Fixed-length epoch segmentation
5. Standardized model-ready EEG tensor

Outputs:

- `data/processed/epochs/*_epochs.npz`
- `data/processed/preprocessing_logs/*_preprocessing_log.json`
- `outputs/figures/*raw_eeg_plot.png`
- `outputs/figures/*filtered_eeg_plot.png`

## 8. Train EEGNet

The default `configs/training.yaml` is for real OpenNeuro training only and has:

```yaml
synthetic_fallback: false
max_train_epochs_per_subject: 30
max_eval_epochs_per_subject: null
```

```bash
python -m src.training.train_eegnet --config configs/training.yaml
```

Outputs:

- `models/checkpoints/eegnet_fold0_best.pth`
- `outputs/reports/fold0_training_history.csv`
- `outputs/reports/fold0_training_summary.json`
- `outputs/figures/fold0_training_curve.png`

The model returns:

- Binary logits
- AD EEG Pattern probability
- `z_eeg`, shape `[256]`
- L2-normalized embedding

For local smoke testing without the dataset, use:

```bash
python -m src.training.train_eegnet --config configs/training_synthetic_test.yaml
```

## 9. Evaluate Subject-Level Metrics

```bash
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml
```

Validate channel order and epoch tensor shapes:

```bash
python -m src.evaluation.validate_preprocessed_shapes --config configs/training.yaml
```

Aggregate all five folds after training checkpoints are available:

```bash
python -m src.evaluation.evaluate_subject_level --config configs/training.yaml --all-folds
```

Outputs:

- `outputs/reports/phase1_metrics.json`
- `outputs/reports/phase1_crossval_metrics.json`
- `outputs/reports/fold0_subject_predictions.csv`
- `outputs/figures/fold0_confusion_matrix.png`
- `outputs/figures/fold0_roc_curve.png`
- `outputs/figures/fold0_embedding_pca.png`
- `outputs/figures/fold0_embedding_tsne.png` when enough subjects are available

Use subject-level metrics as the primary result. Epoch-level scores can be useful for debugging, but they are not the final evaluation because many epochs come from the same subject.

Important metrics:

- AUC-ROC: best overall ranking metric for AD vs HC.
- F1-score: useful when AD/HC class counts are not perfectly balanced.
- Recall/sensitivity: important for detecting Alzheimer's EEG Pattern subjects.
- Precision: shows how reliable AD-positive predictions are.
- Accuracy: useful, but do not trust it alone.
- Confusion matrix: shows false positives and missed AD cases.
- Cross-validation mean/std: checks whether results are stable across folds.
- Embedding silhouette: optional check for whether `z_eeg` separates AD/HC embeddings.

The metric implementation is in `src/evaluation/metrics.py`. In the JSON output, sensitivity is reported as `recall`.

Local files to inspect after evaluation:

```text
outputs/reports/phase1_metrics.json
outputs/reports/phase1_crossval_metrics.json
outputs/reports/fold0_subject_predictions.csv
outputs/figures/fold0_confusion_matrix.png
outputs/figures/fold0_roc_curve.png
outputs/figures/fold0_embedding_pca.png
```

For final reporting, prefer all-fold results from `phase1_crossval_metrics.json` when all checkpoints are trained.

## 9.1 Check Overfitting After Training

After each training run, inspect:

```text
outputs/figures/fold0_training_curve.png
outputs/reports/fold0_training_history.csv
```

The training curve includes:

- Train vs validation loss
- Train vs validation subject accuracy
- Train vs validation subject F1
- Train vs validation subject AUC

Common overfitting signs:

- Training loss keeps falling while validation loss rises.
- Training accuracy/F1/AUC becomes high while validation metrics stay flat or drop.
- A large train-validation metric gap appears after a few epochs.

If this happens, try stronger regularization, lower learning rate, fewer epochs, earlier stopping, more conservative preprocessing, or more subjects/folds before trusting the result.

## 10. Predict One Subject And Generate JSON

```bash
python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-001
```

Outputs:

- `outputs/reports/sub-001_prediction_report.json`
- `outputs/figures/sub-001_probability_bar_chart.png`
- `outputs/figures/sub-001_epoch_probability_distribution.png`
- `models/embeddings/sub-001_z_eeg.npy`

The JSON includes:

- Dataset metadata
- Input metadata
- Preprocessing summary
- Subject-level prediction
- Confidence interpretation
- Epoch probability summary
- Probability bar chart data
- `z_eeg` shape and L2 norm
- Clinical disclaimer

## 11. Modal/Local Post-Training Smoke Test

After training on Modal, run this before presentation or submission:

```bash
python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-001 --fold 0
```

For all trained folds:

```bash
python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-001 --all-folds
```

This script does not start training. It checks:

- Label file and AD/HC labels
- Subject split file and no subject leakage
- Processed epoch files
- Fold checkpoint exists
- Subject-level evaluation runs with `auto_train=False`
- Prediction JSON report is generated
- `z_eeg_shape = [256]`, `l2_norm ~= 1.0`, and `availability_flag = 1`
- Clinical disclaimer is present
- Visual artifact files exist
- Gradio app can be built

It writes:

- `outputs/reports/modal_post_train_check.json`

For local synthetic smoke tests only, add:

```bash
python -m src.evaluation.modal_post_train_check --config configs/training.yaml --subject_id sub-synth-hc-000 --fold 0 --allow-synthetic
```

Keep normal local unit tests separate:

```bash
pytest -q
```

## 12. Launch Gradio Demo

```bash
python -m app.gradio_app
```

The Gradio demo expects a trained checkpoint. It will show a clear error if
`models/checkpoints/eegnet_fold0_best.pth` is missing instead of starting a
training run inside the web app process.

If the demo was already running before code or config changes, stop it and start it again. Gradio launch settings and allowed file paths are only applied when the process starts.

The app can display visual files from local project outputs and Modal artifact paths. It also copies generated figures into:

```text
outputs/gradio_cache/<subject_id>/
```

The demo shows:

- Subject dropdown
- Raw EEG plot
- Filtered EEG plot
- Clean/rejected epoch summary
- Signal quality
- Prediction: Healthy Control / Alzheimer's EEG Pattern
- Confidence score
- Probability chart
- Epoch probability distribution
- `z_eeg` shape and L2 norm
- Clinical disclaimer
- Full JSON report

## 13. Notebook Runner Style

Notebooks are intentionally thin. They call scripts rather than storing model code.

Example cells:

```python
!python -m src.data.prepare_labels --config configs/preprocessing.yaml
```

```python
!python -m src.training.train_eegnet --config configs/training.yaml
```

```python
!python -m src.evaluation.evaluate_subject_level --config configs/training.yaml
```

```python
!python -m src.inference.predict_subject --config configs/training.yaml --subject_id sub-001
```

## 14. Modal Notebook Workflow

Recommended for Phase 1:

- Code path: `/root/eeg_ad_hc_phase1/`
- Dataset volume: `/mnt/eeg-dataset-vol/`
- Artifacts volume: `/mnt/eeg-artifacts-vol/`

Create volumes:

```bash
modal volume create eeg-dataset-vol
modal volume create eeg-artifacts-vol
```

Suggested Modal storage:

```text
/mnt/eeg-dataset-vol/
├── raw/openneuro_ds004504/
└── processed/

/mnt/eeg-artifacts-vol/
├── checkpoints/
├── embeddings/
├── outputs/
└── reports/
```

Update `configs/training.yaml` paths in Modal:

```yaml
paths:
  dataset_root: /mnt/eeg-dataset-vol/raw/openneuro_ds004504
  labels_dir: /mnt/eeg-artifacts-vol/labels
  splits_dir: /mnt/eeg-artifacts-vol/splits
  processed_epochs_dir: /mnt/eeg-dataset-vol/processed/epochs
  preprocessing_logs_dir: /mnt/eeg-dataset-vol/processed/preprocessing_logs
  checkpoints_dir: /mnt/eeg-artifacts-vol/checkpoints
  final_model_dir: /mnt/eeg-artifacts-vol/final
  embeddings_dir: /mnt/eeg-artifacts-vol/embeddings
  reports_dir: /mnt/eeg-artifacts-vol/reports
  figures_dir: /mnt/eeg-artifacts-vol/figures
  logs_dir: /mnt/eeg-artifacts-vol/logs
```

GPU recommendation:

- Use `L40S` if available.
- T4, L4, or A10 are acceptable lower-cost options.
- A100 is not required for Phase 1 EEGNet unless later CWT/Transformer experiments become large.

## 15. Tests

```bash
pytest
```

Tests verify:

- No subject leakage across folds
- EEGNet output shape
- `z_eeg` L2 normalization
- Required JSON report keys and safe wording
- Epoching and amplitude rejection helper behavior

## 16. Acceptance Criteria

Phase 1 is complete when one subject/sample can run through:

```text
Load EEG -> preprocess -> clean epochs -> train/infer EEGNet -> subject-level prediction -> z_eeg -> JSON report
```

Required evidence:

- FTD excluded
- Train/test split is subject-level
- No epoch leakage
- Metrics are subject-level
- `z_eeg_shape = [256]`
- `l2_norm ~= 1.0`
- `availability_flag = 1`
- Clinical disclaimer is present
