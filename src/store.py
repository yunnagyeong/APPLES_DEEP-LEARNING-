"""예시 풀 로드 + 임베딩 색인 관리.

examples.jsonl (사람이 채우는 원본 데이터)과 index_cache.json (임베딩 캐시)을 분리해서,
예시를 추가/수정할 때마다 전체를 재임베딩하지 않고 바뀐 것만 새로 계산하도록 합니다.
"""

import hashlib
import json
import sys

from . import config
from .embeddings import embed_text


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_examples() -> list[dict]:
    """data/examples/examples.jsonl을 읽어 리스트로 반환합니다."""
    if not config.EXAMPLES_PATH.exists():
        print(f"[store] {config.EXAMPLES_PATH} 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    examples = []
    with open(config.EXAMPLES_PATH, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[store] {line_no}번째 줄 JSON 파싱 실패: {e}", file=sys.stderr)
                sys.exit(1)
    return examples


def _load_cache() -> dict:
    if not config.INDEX_CACHE_PATH.exists():
        return {}
    with open(config.INDEX_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_cache(cache: dict) -> None:
    config.INDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.INDEX_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def build_index() -> None:
    """examples.jsonl을 기준으로 임베딩 캐시를 최신 상태로 만듭니다.

    이미 같은 텍스트로 임베딩해둔 예시는 재호출하지 않아 API 사용량을 아낍니다.
    예시를 추가/수정한 뒤에는 이 함수(또는 scripts/build_index.py)를 다시 실행해야 합니다.
    """
    examples = read_examples()
    cache = _load_cache()
    total = len(examples)
    print(f"[store] 총 {total}개 예시 확인, 색인 시작")

    new_count = 0
    for i, ex in enumerate(examples, start=1):
        ex_id = ex["id"]
        h = _text_hash(ex["text"])
        cached = cache.get(ex_id)
        if cached is not None and cached.get("text_hash") == h:
            continue  # 변경 없음, 재사용
        embedding = embed_text(ex["text"], task_type="retrieval_document")
        cache[ex_id] = {"text_hash": h, "embedding": embedding}
        new_count += 1
        print(f"[store] ({i}/{total}) 임베딩 계산: {ex_id} ({ex.get('type', '?')})")
        if i % 10 == 0:
            _save_cache(cache)  # 중간에 실패해도 여기까지는 재사용 가능하도록 주기적으로 저장

    _save_cache(cache)
    print(
        f"[store] 완료. 전체 {len(examples)}개 중 {new_count}개 새로 임베딩, "
        f"{len(examples) - new_count}개 캐시 재사용."
    )


def load_indexed_examples() -> tuple[list[dict], list[list[float]]]:
    """검색에 쓸 (예시 메타데이터 리스트, 임베딩 벡터 리스트)를 반환합니다.

    캐시에 없는 예시가 있으면 색인이 오래된 것이므로, build_index()를 먼저 실행하라고 안내합니다.
    """
    examples = read_examples()
    cache = _load_cache()

    vectors = []
    missing = []
    for ex in examples:
        cached = cache.get(ex["id"])
        h = _text_hash(ex["text"])
        if cached is None or cached.get("text_hash") != h:
            missing.append(ex["id"])
            continue
        vectors.append(cached["embedding"])

    if missing:
        print(
            "[store] 다음 예시가 아직 색인되지 않았습니다: "
            f"{', '.join(missing)}\n"
            "        `python scripts/build_index.py`를 먼저 실행해주세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    return examples, vectors
