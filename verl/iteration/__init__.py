"""Iteration-level co-evolution primitives.

The training stack remains responsible for individual compute-heavy phases;
this package owns the versioned contracts that connect those phases.
"""

from .core import (
    Candidate,
    EvidenceItem,
    IterationState,
    Rubric,
    RubricEvaluation,
    Skill,
    StateStore,
    VerifyResult,
    initial_rubrics,
    initial_skills,
)

__all__ = [
    "Candidate",
    "EvidenceItem",
    "IterationState",
    "Rubric",
    "RubricEvaluation",
    "Skill",
    "StateStore",
    "VerifyResult",
    "initial_rubrics",
    "initial_skills",
]
