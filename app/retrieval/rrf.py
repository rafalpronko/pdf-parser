"""Reciprocal Rank Fusion (RRF) for combining multiple ranked lists."""

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list],
    weights: list[float] | None = None,
    k: int = 60,
    id_fn=lambda item: item.chunk_id,
) -> dict[str, float]:
    """Compute RRF scores from multiple ranked lists.

    RRF score for item d: sum over all rankings of weight_i / (k + rank(d))

    Args:
        ranked_lists: List of ranked result lists
        weights: Optional per-list weights (default: equal weights of 1.0)
        k: RRF smoothing constant (default 60)
        id_fn: Function to extract unique ID from an item

    Returns:
        Dict mapping item ID to RRF score
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores: dict[str, float] = defaultdict(float)

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked_list):
            item_id = id_fn(item)
            rrf_scores[item_id] += weight / (k + rank + 1)

    return dict(rrf_scores)
