from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.inference.predict_subject import predict_subject
from src.utils.config_loader import load_config, resolve_data_file, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)
_CONFIG_PATH = "configs/training.yaml"


def checkpoint_exists() -> bool:
    try:
        cfg = load_config(_CONFIG_PATH)
        checkpoints_dir = resolve_path(cfg["paths"]["checkpoints_dir"])
        return any(checkpoints_dir.glob("eegnet_fold*_best.pth"))
    except Exception:
        return False


def _subject_split_membership(subject_id: str) -> dict:
    try:
        cfg = load_config(_CONFIG_PATH)
        split_file = resolve_data_file(cfg, "split_file", "splits_dir")
        active_fold = int(cfg["data"].get("active_fold", 0))
        if not split_file.exists():
            return {"active_fold": active_fold, "split": "unknown"}
        import pandas as pd

        splits = pd.read_csv(split_file)
        match = splits[
            (splits["fold"].astype(int) == active_fold) &
            (splits["subject_id"].astype(str) == str(subject_id))
        ]
        split = str(match.iloc[0]["split"]) if len(match) else "unknown"
        return {"active_fold": active_fold, "split": split}
    except Exception:
        return {"active_fold": None, "split": "unknown"}


def _copy_visual_to_gradio_cache(path: str | None, subject_id: str) -> str | None:
    if not path:
        return None
    src_path = Path(path)
    if not src_path.exists():
        return None
    cache_dir = Path("outputs") / "gradio_cache" / str(subject_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst_path = cache_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    return str(dst_path.resolve())


def run_demo_subject(subject_id: str):
    """Run the EEG analysis pipeline and return Gradio-friendly outputs."""
    if not checkpoint_exists():
        message = (
            f"No trained model checkpoint found for subject {subject_id}.\n\n"
            "Run training first:\n"
            "  python -m src.training.train_eegnet --config configs/training.yaml --fold 0\n\n"
            "Then restart the Gradio demo."
        )
        return None, None, None, None, message, json.dumps({"error": "No checkpoint found."}, indent=2)

    try:
        report = predict_subject(subject_id, auto_train=False)
    except Exception as exc:
        LOGGER.exception("Demo pipeline failed for subject %s", subject_id)
        no_image = None
        error_summary = (
            f"Analysis failed for {subject_id}.\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            "Check that preprocessing has been run and a model checkpoint exists.\n"
            "See logs for the full traceback."
        )
        return no_image, no_image, no_image, no_image, error_summary, "{}"

    pred = report["subject_level_prediction"]
    emb = report["embedding_output"]
    prep = report["preprocessing_summary"]
    visuals = report["visual_outputs"]
    meta = report["input_metadata"]
    demo_split = _subject_split_membership(subject_id)

    # Gradio only allows serving files from cwd or temp (unless allowed_paths is set).
    # Copy Modal volume artifacts into a cwd-local cache for reliable display.
    raw_plot_path = _copy_visual_to_gradio_cache(visuals.get("raw_eeg_plot"), subject_id)
    filtered_plot_path = _copy_visual_to_gradio_cache(visuals.get("filtered_eeg_plot"), subject_id)
    prob_chart_path = _copy_visual_to_gradio_cache(visuals.get("probability_bar_chart"), subject_id)
    epoch_dist_path = _copy_visual_to_gradio_cache(visuals.get("epoch_probability_distribution"), subject_id)

    synthetic_warning = ""
    if meta.get("synthetic_data"):
        synthetic_warning = "\nWARNING: SYNTHETIC DATA - not a real EEG recording."

    summary = (
        f"Subject:    {subject_id}{synthetic_warning}\n"
        f"Demo split: fold {demo_split['active_fold']} / {demo_split['split']}\n"
        f"Real data:  {bool(meta.get('real_data', not meta.get('synthetic_data', False)))}\n"
        f"Prediction: {pred['prediction']}\n"
        f"Confidence: {pred['subject_level_confidence']:.2f}\n"
        f"Risk level: {pred['risk_level']}\n"
        f"Clean epochs: {prep['clean_epochs_used']} / {prep['total_epochs_generated']}\n"
        f"Signal quality: {prep['signal_quality']}\n"
        f"z_eeg shape: {emb['z_eeg_shape']}, L2 norm: {emb['l2_norm']}\n"
        f"Embedding consistency: {emb['embedding_consistency']:.3f}"
    )
    return (
        raw_plot_path,
        filtered_plot_path,
        prob_chart_path,
        epoch_dist_path,
        summary,
        json.dumps(report, indent=2),
    )
