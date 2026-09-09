"""rag_writing_cleaned_final.jsonl(원본 문서)을 data/examples/examples.jsonl 스키마로 변환합니다.

문체 예시로 쓰기에 적절한 단위를 만들기 위해, 단순 글자 수 기준(chunked_700)이 아니라
문단 경계(빈 줄)를 기준으로 자릅니다. 원본은 줄바꿈이 워드/PDF 추출 과정에서 생긴 강제
개행(soft wrap)인 경우가 많아 문단 내부 개행은 공백으로 합칩니다. 너무 짧은 문단(불릿 한
줄 등)은 다음 문단과 합치고, 너무 긴 문단(에세이형 글)은 문장 경계에서 다시 나눠서, 완결된
문장으로 시작·끝나는 200~500자 안팎의 조각을 만듭니다. (스코프: 이 프로젝트의 소스 하나뿐이라
build_index.py처럼 범용화하지 않고 일회성 변환 스크립트로 둡니다.)

사용법: python scripts/convert_data_to_examples.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

SOURCE_PATH = config.ROOT_DIR / "data" / "rag_writing_cleaned_final.jsonl"

TARGET_MIN = 150
TARGET_MAX = 500
SENTENCE_SPLIT_THRESHOLD = 550

TYPE_RULES = [
    ("리포트", "report"),
    ("기획서", "proposal"),
    ("에세이", "essay"),
    ("활동지", "worksheet"),
    ("과제", "assignment"),
]


def infer_type(title: str) -> str:
    for keyword, type_name in TYPE_RULES:
        if keyword in title:
            return type_name
    return "note"


def normalize_paragraph(paragraph: str) -> str:
    return re.sub(r"\s*\n\s*", " ", paragraph).strip()


def split_into_sentences(paragraph: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in sentences if s.strip()]


def atomic_pieces(text: str) -> list[str]:
    """빈 줄 기준으로 문단을 나누고, 너무 긴 문단은 문장 단위로 다시 쪼갭니다."""
    raw_paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces = []
    for raw in raw_paragraphs:
        paragraph = normalize_paragraph(raw)
        if len(paragraph) <= SENTENCE_SPLIT_THRESHOLD:
            pieces.append(paragraph)
        else:
            pieces.extend(split_into_sentences(paragraph))
    return pieces


def merge_pieces(pieces: list[str]) -> list[str]:
    """작은 조각은 이어 붙이고, TARGET_MAX를 넘기기 직전에 새 청크를 시작합니다."""
    chunks = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer} {piece}".strip() if buffer else piece
        if len(buffer) >= TARGET_MIN and len(candidate) > TARGET_MAX:
            chunks.append(buffer)
            buffer = piece
        else:
            buffer = candidate
    if buffer:
        if chunks and len(buffer) < TARGET_MIN:
            chunks[-1] = f"{chunks[-1]} {buffer}".strip()
        else:
            chunks.append(buffer)
    return chunks


def main() -> None:
    if not SOURCE_PATH.exists():
        print(f"[convert] 원본 파일을 찾을 수 없습니다: {SOURCE_PATH}", file=sys.stderr)
        sys.exit(1)

    examples = []
    with open(SOURCE_PATH, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    for doc in docs:
        title = doc["title"]
        doc_type = infer_type(title)
        pieces = atomic_pieces(doc["text"])
        chunks = merge_pieces(pieces)
        for i, chunk in enumerate(chunks, start=1):
            examples.append(
                {
                    "id": f"{doc['id']}_{i:02d}",
                    "type": doc_type,
                    "topic": title,
                    "text": chunk,
                }
            )

    out_path = config.EXAMPLES_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    lengths = [len(ex["text"]) for ex in examples]
    print(f"[convert] 문서 {len(docs)}개 -> 예시 {len(examples)}개 생성")
    print(f"[convert] 길이 범위: min={min(lengths)} max={max(lengths)} "
          f"평균={sum(lengths) // len(lengths)}")
    print(f"[convert] 저장 위치: {out_path}")


if __name__ == "__main__":
    main()
