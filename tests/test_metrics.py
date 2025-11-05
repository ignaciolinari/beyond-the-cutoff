from beyond_the_cutoff.evaluation.metrics import (
    citation_precision,
    citation_recall,
    evaluate_citations,
    grounded_fraction,
    normalize_contexts,
)


def test_evaluate_citations_metrics() -> None:
    contexts = normalize_contexts(
        ["Paris is the capital of France.", "Berlin is the capital of Germany."]
    )
    answer = "The capital of France is Paris. [1]"
    metrics = evaluate_citations(answer, contexts)

    assert metrics["referenced"] == [1]
    assert metrics["missing"] == [2]
    assert metrics["extra"] == []
    assert citation_precision(metrics) == 1.0
    assert citation_recall(metrics, total_contexts=len(contexts)) == 0.5


def test_grounded_fraction_overlap() -> None:
    contexts = normalize_contexts(["Paris is the capital of France."])
    grounded = grounded_fraction("Paris remains the capital of France.", contexts)
    assert 0.0 < grounded <= 1.0

    no_ground = grounded_fraction("Tokyo leads technology innovations.", contexts)
    assert no_ground < grounded
