"""Local Gradio shell for the P1_CXR E6 research-demo workflow."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr
from demo_core.cases import CaseStore, discover_asset_paths
from demo_core.gradcam import generate_gradcam_overlay
from demo_core.model_adapter import (
    CANONICAL_CLASSES,
    CHECKPOINT_ENVIRONMENT_VARIABLE,
    E6ModelAdapter,
)
from demo_core.presentation import (
    DISPLAY_NAMES,
    WORKLIST_HEADERS,
    case_options,
    detail_markdown,
    worklist_rows,
)
from demo_core.priority import DEFAULT_HIGH_REVIEW_THRESHOLD

DEMO_ROOT = Path(__file__).resolve().parent
ASSET_DIRECTORY = DEMO_ROOT / "assets" / "demo_cases"

CSS = """
.gradio-container { max-width: 1440px !important; background: #f7fafc; }
#demo-title { margin-bottom: 0.2rem; }
.status-good { color: #087443; font-weight: 650; }
.status-warn { color: #9a5300; font-weight: 650; }
.disclaimer { color: #4a5568; font-size: 0.88rem; margin-top: 0.7rem; }
"""


def status_markdown(adapter: E6ModelAdapter) -> str:
    if adapter.is_connected:
        return f"<span class='status-good'>● Model connected</span> — `{adapter.checkpoint_path}`"
    return f"<span class='status-warn'>● Model unavailable</span> — {adapter.status.message}"


def initial_store(adapter: E6ModelAdapter) -> CaseStore:
    store = CaseStore()
    store.add_paths(discover_asset_paths(ASSET_DIRECTORY), adapter)
    return store


def build_demo(adapter: E6ModelAdapter) -> gr.Blocks:
    store_at_startup = initial_store(adapter)

    def add_cases(paths: list[str] | None, high_threshold: float, store: CaseStore):
        active_store = store or CaseStore()
        added, notes = active_store.add_paths(paths, adapter)
        options = case_options(active_store)
        selected = options[0] if options else None
        message = f"Added {added} anonymous case(s)." if added else "No supported CXR images were added."
        if notes:
            message += " " + " ".join(notes)
        return (
            active_store,
            worklist_rows(active_store, high_threshold),
            gr.Dropdown(choices=options, value=selected),
            message,
        )

    def refresh_for_threshold(store: CaseStore, high_threshold: float, selected_case_id: str | None):
        case = (store or CaseStore()).get(selected_case_id)
        return (
            worklist_rows(store or CaseStore(), high_threshold),
            case.image if case else None,
            detail_markdown(case, high_threshold),
        )

    def show_detail(store: CaseStore, high_threshold: float, case_id: str | None):
        case = (store or CaseStore()).get(case_id)
        attention_status = (
            "Choose an E6 target class and select **Generate AI Attention**."
            if case and case.prediction and adapter.is_connected
            else "AI Attention is unavailable until a valid full-training E6 checkpoint is connected."
        )
        return case.image if case else None, detail_markdown(case, high_threshold), case.image if case else None, None, attention_status

    def select_worklist(store: CaseStore, high_threshold: float, event: gr.SelectData):
        rows = worklist_rows(store or CaseStore(), high_threshold)
        row_index = event.index[0] if isinstance(event.index, tuple) else event.index
        if not isinstance(row_index, int) or row_index < 0 or row_index >= len(rows):
            return gr.Dropdown()
        return gr.Dropdown(value=rows[row_index][0])

    def generate_attention(store: CaseStore, case_id: str | None, target_display_name: str):
        case = (store or CaseStore()).get(case_id)
        if case is None:
            return None, None, "Select a case before generating AI Attention."
        if not adapter.is_connected or case.prediction is None:
            return case.image, None, "AI Attention is unavailable: a valid full-training E6 checkpoint is required."
        target_class = next(name for name, display in DISPLAY_NAMES.items() if display == target_display_name)
        try:
            overlay = generate_gradcam_overlay(adapter, case.image, target_class)
        except Exception as error:  # noqa: BLE001 - UI must report unexpected local inference errors.
            return case.image, None, f"AI Attention could not be generated: {error}"
        return case.image, overlay, (
            f"AI Attention Visualization for **{target_display_name}**. "
            "This visualization indicates image regions that influenced the model output. "
            "It is not a confirmed lesion localization."
        )

    with gr.Blocks(css=CSS, title="CXR AI Review & Attention Explorer") as demo:
        store_state = gr.State(store_at_startup)
        gr.Markdown("# CXR AI Review & Attention Explorer", elem_id="demo-title")
        gr.Markdown(
            "Anonymous CXR worklist for a research presentation. "
            "AI Review Priority is a model-derived display score, not clinical urgency."
        )
        gr.Markdown(status_markdown(adapter))

        with gr.Row():
            with gr.Column(scale=3):
                upload = gr.File(
                    label="Add de-identified CXR cases",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
            with gr.Column(scale=1):
                add_button = gr.Button("Add to worklist", variant="primary")
                upload_result = gr.Markdown("No images have been added in this session.")

        gr.Markdown("## CXR AI Review Worklist")
        high_threshold = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=DEFAULT_HIGH_REVIEW_THRESHOLD,
            step=0.01,
            label="AI Review Threshold (HIGH REVIEW PRIORITY)",
        )
        gr.Markdown("`Research demo threshold — not a clinical urgency threshold`", elem_classes=["disclaimer"])
        worklist = gr.Dataframe(
            value=worklist_rows(store_at_startup, DEFAULT_HIGH_REVIEW_THRESHOLD),
            headers=WORKLIST_HEADERS,
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            label="Sorted by AI Review Priority Score: 1 − P(Normal)",
        )
        case_selector = gr.Dropdown(
            choices=case_options(store_at_startup),
            value=None,
            label="Select an anonymous case",
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Case Detail")
                detail_image = gr.Image(label="Original CXR", type="pil", interactive=False)
            with gr.Column():
                detail = gr.Markdown("Select an anonymous case from the worklist to view its analysis.")

        gr.Markdown("## AI Attention Explorer")
        gr.Markdown("Original ←→ AI Attention (side-by-side fallback for reliable local presentation)")
        target_class = gr.Radio(
            choices=[DISPLAY_NAMES[name] for name in CANONICAL_CLASSES],
            value=DISPLAY_NAMES["BACTERIAL_PNEUMONIA"],
            label="AI Attention target class",
        )
        attention_button = gr.Button("Generate AI Attention", variant="secondary")
        with gr.Row():
            attention_original = gr.Image(label="Original", type="pil", interactive=False)
            attention_overlay = gr.Image(label="AI Attention Visualization", type="pil", interactive=False)
        attention_status = gr.Markdown("AI Attention requires a valid full-training E6 checkpoint and a selected case.")

        gr.Markdown(
            "This AI demo is intended for research and educational purposes only and does not replace clinical diagnosis.  \\n"
            "This prototype is limited to model predictions for Normal, Bacterial Pneumonia, and Viral Pneumonia.",
            elem_classes=["disclaimer"],
        )

        add_button.click(
            add_cases,
            inputs=[upload, high_threshold, store_state],
            outputs=[store_state, worklist, case_selector, upload_result],
        )
        high_threshold.change(
            refresh_for_threshold,
            inputs=[store_state, high_threshold, case_selector],
            outputs=[worklist, detail_image, detail],
        )
        worklist.select(select_worklist, inputs=[store_state, high_threshold], outputs=[case_selector])
        case_selector.change(
            show_detail,
            inputs=[store_state, high_threshold, case_selector],
            outputs=[detail_image, detail, attention_original, attention_overlay, attention_status],
        )
        attention_button.click(
            generate_attention,
            inputs=[store_state, case_selector, target_class],
            outputs=[attention_original, attention_overlay, attention_status],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help=f"Override {CHECKPOINT_ENVIRONMENT_VARIABLE} for this launch.")
    parser.add_argument("--host", default="127.0.0.1", help="Local bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=None, help="Optional local port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint:
        os.environ[CHECKPOINT_ENVIRONMENT_VARIABLE] = args.checkpoint
    adapter = E6ModelAdapter.from_environment()
    build_demo(adapter).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
