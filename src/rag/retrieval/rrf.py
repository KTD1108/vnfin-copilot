"""Thuật toán Reciprocal Rank Fusion (RRF) & Gom nhóm theo Parent ID."""

from collections import defaultdict

from config import RRF_K_CONSTANT
from rag.schemas import RankedCandidate


def reciprocal_rank_fusion(
    ranked_lists: list[list[RankedCandidate]],
    k: int = RRF_K_CONSTANT,
    deduplicate_by_parent: bool = True,
) -> list[RankedCandidate]:
    """Hợp nhất nhiều danh sách xếp hạng (Dense, Sparse BM25) bằng công thức RRF.

    Công thức: RRF_Score(d) = sum(1 / (k + rank_i(d))) với rank bắt đầu từ 1.
    """
    if k <= 0:
        raise ValueError("Hằng số k của RRF phải lớn hơn 0.")

    rrf_scores: dict[str, float] = defaultdict(float)
    candidate_map: dict[str, RankedCandidate] = {}

    for candidates in ranked_lists:
        for rank_idx, item in enumerate(candidates, start=1):
            rrf_scores[item.chunk_id] += 1.0 / (k + rank_idx)
            if item.chunk_id not in candidate_map:
                candidate_map[item.chunk_id] = item

    fused = [
        RankedCandidate(
            chunk_id=cid,
            parent_id=candidate_map[cid].parent_id,
            score=score,
            metadata=candidate_map[cid].metadata,
        )
        for cid, score in sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    ]

    if not deduplicate_by_parent:
        return fused

    return _deduplicate_parents(fused)


def _deduplicate_parents(candidates: list[RankedCandidate]) -> list[RankedCandidate]:
    """Giữ lại ứng viên có điểm RRF cao nhất cho mỗi parent_id."""
    seen_parents: set[str] = set()
    unique_results: list[RankedCandidate] = []

    for item in candidates:
        if item.parent_id in seen_parents:
            continue
        seen_parents.add(item.parent_id)
        unique_results.append(item)

    return unique_results
