"""Pure formatting helpers for the Gradio research-demo UI."""
from __future__ import annotations

from .cases import CaseStore, DemoCase
from .priority import DEFAULT_HIGH_REVIEW_THRESHOLD, ReviewPriority, assess_priority

DISPLAY_NAMES = {
    "NORMAL": "Normal",
    "BACTERIAL_PNEUMONIA": "Bacterial Pneumonia",
    "VIRAL_PNEUMONIA": "Viral Pneumonia",
}
WORKLIST_HEADERS = [
    "Case",
    "AI Review Priority",
    "AI Review Priority Score",
    "Predicted Class",
    "Confidence",
]


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def worklist_rows(store: CaseStore, high_threshold: float) -> list[list[str]]:
    rows: list[tuple[tuple[int, float], list[str]]] = []
    for case in store.cases:
        if case.prediction is None:
            row = [case.case_id, ReviewPriority.UNAVAILABLE.value, "—", "—", "—"]
            rows.append(((1, 0.0), row))
            continue
        priority = assess_priority(
            case.prediction.probabilities["NORMAL"],
            high_threshold=high_threshold,
        )
        row = [
            case.case_id,
            priority.group.value,
            percent(priority.score),
            DISPLAY_NAMES[case.prediction.predicted_class],
            percent(case.prediction.confidence),
        ]
        rows.append(((0, -priority.score), row))
    return [row for _, row in sorted(rows, key=lambda item: (item[0], item[1][0]))]


def case_options(store: CaseStore) -> list[str]:
    return [case.case_id for case in store.cases]


def detail_markdown(case: DemoCase | None, high_threshold: float = DEFAULT_HIGH_REVIEW_THRESHOLD) -> str:
    if case is None:
        return "Select an anonymous case from the worklist to view its analysis."
    if case.prediction is None:
        note = case.inference_note or "No prediction is available for this case."
        return f"## {case.case_id}\n\n**Model unavailable**\n\n{note}"

    prediction = case.prediction
    priority = assess_priority(prediction.probabilities["NORMAL"], high_threshold=high_threshold)
    probabilities = "\n".join(
        f"- **{DISPLAY_NAMES[name]}:** {percent(prediction.probabilities[name])}"
        for name in ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
    )
    return (
        f"## {case.case_id}\n\n"
        "### Prediction\n"
        f"{probabilities}\n\n"
        f"- **Predicted class:** {DISPLAY_NAMES[prediction.predicted_class]}\n"
        f"- **Prediction confidence:** {percent(prediction.confidence)}\n"
        f"- **AI Review Priority:** {priority.group.value}\n"
        f"- **AI Review Priority Score:** {percent(priority.score)}\n\n"
        "AI Review Priority Score is a model-derived research-demo display value, not clinical urgency."
    )
