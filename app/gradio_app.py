from __future__ import annotations

import os

import gradio as gr

from app.demo_utils import checkpoint_exists, run_demo_subject
from app.subject_options import load_demo_subjects
from src.inference.output_schema import CLINICAL_DISCLAIMER
from src.utils.config_loader import load_config, resolve_path


def _allowed_paths() -> list[str]:
    cfg = load_config("configs/training.yaml")
    path_keys = [
        "figures_dir",
        "reports_dir",
        "logs_dir",
        "processed_epochs_dir",
        "embeddings_dir",
    ]
    paths = [str(resolve_path(cfg["paths"][key])) for key in path_keys if key in cfg["paths"]]
    paths.append(str(resolve_path("outputs/gradio_cache")))
    return paths


def build_app() -> gr.Blocks:
    subjects = load_demo_subjects()
    checkpoint_ok = checkpoint_exists()
    with gr.Blocks(title="EEG AD/HC Phase 1 Research Demo") as demo:
        gr.Markdown("# EEG AD/HC Phase 1 Research Demo")
        if not checkpoint_ok:
            gr.Textbox(
                value=(
                    "No trained model checkpoint found.\n"
                    "Run training before using this demo:\n\n"
                    "python -m src.training.train_eegnet --config configs/training.yaml --fold 0"
                ),
                label="Setup required",
                interactive=False,
                lines=5,
            )
        subject = gr.Dropdown(
            choices=subjects,
            value=subjects[0],
            label="Select sample subject",
        )
        run = gr.Button("Run EEG Analysis", interactive=checkpoint_ok)
        with gr.Row():
            raw_plot = gr.Image(label="Raw EEG plot", type="filepath")
            filtered_plot = gr.Image(label="Filtered EEG plot", type="filepath")
        with gr.Row():
            prob_chart = gr.Image(label="Class probability bar chart", type="filepath")
            epoch_dist = gr.Image(label="Epoch probability distribution", type="filepath")
        summary = gr.Textbox(label="Subject-level result", lines=7)
        disclaimer = gr.Textbox(value=CLINICAL_DISCLAIMER, label="Clinical disclaimer", interactive=False)
        report = gr.Code(label="Full JSON report", language="json")
        run.click(run_demo_subject, inputs=subject, outputs=[raw_plot, filtered_plot, prob_chart, epoch_dist, summary, report])
    return demo


if __name__ == "__main__":
    build_app().launch(
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        allowed_paths=_allowed_paths(),
    )
