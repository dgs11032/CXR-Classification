"""Research-demo priority calculation for the canonical E6 classes."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

LOW_REVIEW_THRESHOLD = 0.30
DEFAULT_HIGH_REVIEW_THRESHOLD = 0.70


class ReviewPriority(str, Enum):
    HIGH = "HIGH REVIEW PRIORITY"
    REVIEW = "REVIEW"
    LOW = "LOW REVIEW PRIORITY"
    UNAVAILABLE = "MODEL UNAVAILABLE"


@dataclass(frozen=True)
class PriorityAssessment:
    score: float
    group: ReviewPriority


def abnormality_score(normal_probability: float) -> float:
    """Return the E6 research-demo score: 1 - P(NORMAL)."""
    if not 0.0 <= normal_probability <= 1.0:
        raise ValueError("NORMAL probability must be within [0, 1].")
    return 1.0 - normal_probability


def assess_priority(
    normal_probability: float,
    *,
    high_threshold: float = DEFAULT_HIGH_REVIEW_THRESHOLD,
    low_threshold: float = LOW_REVIEW_THRESHOLD,
) -> PriorityAssessment:
    """Assign a display-only AI review group; this is not clinical urgency."""
    if not 0.0 <= low_threshold <= high_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= low <= high <= 1.")
    score = abnormality_score(normal_probability)
    if score >= high_threshold:
        group = ReviewPriority.HIGH
    elif score >= low_threshold:
        group = ReviewPriority.REVIEW
    else:
        group = ReviewPriority.LOW
    return PriorityAssessment(score=score, group=group)


def priority_sort_key(assessment: PriorityAssessment | None) -> tuple[int, float]:
    """Keep unavailable cases last, then show higher review scores first."""
    if assessment is None:
        return (1, 0.0)
    return (0, -assessment.score)
