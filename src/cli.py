"""커맨드라인 진입점.

사용법:
    python -m src.cli rewrite --text "문장 또는 문단" [--k 4]
    python -m src.cli rewrite --file path/to/document.txt [--k 4]
"""

import argparse
import sys
from pathlib import Path

from . import config
from .generate import transform_style
from .retrieve import retrieve_similar


def _print_retrieved(examples: list[dict]) -> None:
    print("--- 참고한 예시 ---", file=sys.stderr)
    for ex in examples:
        print(f"  ({ex['score']:.2f}) [{ex.get('type', '?')}] {ex['text'][:40]}", file=sys.stderr)
    print("-------------------", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="개인 문체 반영 RAG 시스템")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rewrite_parser = subparsers.add_parser(
        "rewrite", help="내용은 유지한 채 입력 텍스트를 내 문체로 변환"
    )
    source_group = rewrite_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", help="변환할 문장 또는 문단")
    source_group.add_argument("--file", help="변환할 텍스트 파일 경로")
    rewrite_parser.add_argument("--k", type=int, default=config.DEFAULT_TOP_K)
    rewrite_parser.add_argument(
        "--type",
        dest="type_filter",
        default=None,
        help="이 장르(예: report, essay, proposal, worksheet, note, assignment)의 예시만 참고",
    )

    args = parser.parse_args()

    if args.command == "rewrite":
        if args.file:
            path = Path(args.file)
            if not path.exists():
                print(f"파일을 찾을 수 없습니다: {path}", file=sys.stderr)
                sys.exit(1)
            source_text = path.read_text(encoding="utf-8")
        else:
            source_text = args.text

        print("[cli] (1/2) 유사 예시 검색 중...", file=sys.stderr)
        examples = retrieve_similar(
            source_text[:500], k=args.k, type_filter=args.type_filter
        )  # 앞부분으로 검색 질의 구성
        _print_retrieved(examples)
        print("[cli] (2/2) 생성 모델 호출 중...", file=sys.stderr)
        result = transform_style(source_text, examples)
        print("[cli] 완료", file=sys.stderr)
        print(result)


if __name__ == "__main__":
    main()
