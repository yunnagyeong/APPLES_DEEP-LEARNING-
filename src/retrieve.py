"""코사인 유사도 기반 Top-K 검색.

예시가 수백 개 수준일 때는 별도 벡터 DB(FAISS 등) 없이 numpy 브루트포스로 충분히 빠릅니다.
스터디에서 다룬 어텐션의 "유사도 계산 → 상위 항목 선택" 단계를 그대로 코드로 옮긴 부분입니다.
"""

import numpy as np

from . import config
from .embeddings import embed_text
from .store import load_indexed_examples


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norm @ query_norm


def retrieve_similar(
    query: str, k: int = config.DEFAULT_TOP_K, type_filter: str | None = None
) -> list[dict]:
    """query와 가장 비슷한 내 예시 상위 k개를 유사도 점수와 함께 반환합니다.

    type_filter를 지정하면 그 장르(예: "report", "essay")의 예시만 대상으로 검색합니다.
    일반적인 문장은 특정 장르의 어휘를 담고 있지 않아 의미 검색만으로는 엉뚱한 장르의 예시가
    뽑힐 수 있는데(예: "이번 주차 활동 정리"), 원하는 장르가 이미 정해져 있다면 이 옵션으로
    강제할 수 있습니다.

    반환값은 [{"score": float, **example}, ...] 형태이며 유사도가 높은 순으로 정렬됩니다.
    """
    examples, vectors = load_indexed_examples()
    if type_filter is not None:
        filtered = [(ex, vec) for ex, vec in zip(examples, vectors) if ex.get("type") == type_filter]
        examples = [ex for ex, _ in filtered]
        vectors = [vec for _, vec in filtered]
    if not examples:
        return []

    matrix = np.array(vectors, dtype=np.float32)
    query_vec = np.array(embed_text(query, task_type="retrieval_query"), dtype=np.float32)

    scores = _cosine_similarity(query_vec, matrix)
    top_idx = np.argsort(-scores)[:k]

    return [{"score": float(scores[i]), **examples[i]} for i in top_idx]
