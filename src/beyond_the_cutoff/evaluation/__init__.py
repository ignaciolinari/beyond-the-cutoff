"""Evaluation modules for model comparison and ranking."""

from beyond_the_cutoff.evaluation.elo_ranking import (
    ELOCalculator,
    ELORating,
    PairwiseComparison,
    bootstrap_elo_confidence,
    compute_elo_rankings,
    head_to_head_matrix,
    load_comparisons_from_jsonl,
    save_comparisons_to_jsonl,
    save_leaderboard,
)
from beyond_the_cutoff.evaluation.human_evaluation import (
    AnnotationBatch,
    AnnotationTask,
    HumanAnnotation,
    SamplingStrategy,
    cohens_kappa,
    compute_agreement_stats,
    create_pairwise_tasks,
    export_annotations_for_elo,
    fleiss_kappa,
    human_judge_correlation,
    sample_for_annotation,
)
from beyond_the_cutoff.evaluation.pairwise_judge import (
    MultiJudgeEvaluator,
    MultiJudgeResult,
    PairwiseJudge,
    PairwiseJudgeConfig,
    PairwiseJudgment,
    compute_consensus,
    load_predictions_from_results,
    run_pairwise_evaluation,
)

__all__ = [
    # ELO ranking
    "PairwiseComparison",
    "ELORating",
    "ELOCalculator",
    "bootstrap_elo_confidence",
    "compute_elo_rankings",
    "head_to_head_matrix",
    "load_comparisons_from_jsonl",
    "save_comparisons_to_jsonl",
    "save_leaderboard",
    # Pairwise judge
    "PairwiseJudgeConfig",
    "PairwiseJudgment",
    "PairwiseJudge",
    "MultiJudgeResult",
    "MultiJudgeEvaluator",
    "compute_consensus",
    "load_predictions_from_results",
    "run_pairwise_evaluation",
    # Human evaluation
    "AnnotationTask",
    "HumanAnnotation",
    "AnnotationBatch",
    "SamplingStrategy",
    "sample_for_annotation",
    "create_pairwise_tasks",
    "cohens_kappa",
    "fleiss_kappa",
    "compute_agreement_stats",
    "human_judge_correlation",
    "export_annotations_for_elo",
]
