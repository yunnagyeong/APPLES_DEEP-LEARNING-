"""커맨드라인 진입점.

사용법:
    # 개인화 번역 + 의미 검증 (앱의 조건 C와 같은 파이프라인)
    python -m src.cli translate --user ganghee --text "이 결과는 우연일 수도 있다."
    python -m src.cli translate --user suyoung --file 원문.txt --retries 1

    # 한국어 → 한국어 문체 변환 (초기 기능)
    python -m src.cli rewrite --user ganghee --text "문장 또는 문단" [--k 4]
"""

import argparse
import sys
from pathlib import Path

from . import config


def _print_retrieved(examples: list[dict]) -> None:
    print("--- 참고한 예시 ---", file=sys.stderr)
    for ex in examples:
        print(f"  ({ex['score']:.2f}) [{ex.get('type', '?')}] {ex['text'][:40]}", file=sys.stderr)
    print("-------------------", file=sys.stderr)


def _read_source(args) -> str:
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"파일을 찾을 수 없습니다: {path}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8")
    return args.text


def _add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--user", required=True, help="예시 풀 소유자 (ganghee / nakyung / suyoung)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="입력 문장 또는 문단")
    g.add_argument("--file", help="입력 텍스트 파일 경로")


def main() -> None:
    parser = argparse.ArgumentParser(description="개인 문체 반영 RAG 시스템")
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("translate", help="한국어 → 영어 개인화 번역 + 의미 검증")
    _add_source_args(tp)
    tp.add_argument("--retries", type=int, default=config.DEFAULT_MAX_RETRIES, help="최대 재번역 횟수")

    rp = sub.add_parser("rewrite", help="내용은 유지한 채 입력 텍스트를 내 문체로 변환(한국어)")
    _add_source_args(rp)
    rp.add_argument("--k", type=int, default=config.DEFAULT_TOP_K)
    rp.add_argument("--type", dest="type_filter", default=None,
                    help="이 장르(예: report, essay, proposal, note)의 예시만 참고")

    args = parser.parse_args()
    source_text = _read_source(args)

    if args.command == "translate":
        from .pipeline import _load_or_compute_style_profile, translate_with_verification

        profile = _load_or_compute_style_profile(args.user)
        r = translate_with_verification(source_text, profile, args.user, max_retries=args.retries)
        print(f"[cli] 판정 {r['verdict']} · Meaning {r['meaning_score']} · 시도 {r['attempts']}회"
              f"(채택 {r['best_attempt']}번째) · 판정 모델 {r['judge_model']}", file=sys.stderr)
        for i in r["issues"]:
            print(f"  [{i['severity']}] {i['label_ko']}: {i.get('note', '')}", file=sys.stderr)
        if r["notice"]:
            print(r["notice"].strip(), file=sys.stderr)
        print(r["translation"])

    elif args.command == "rewrite":
        from .generate import transform_style
        from .retrieve import retrieve_similar

        print("[cli] (1/2) 유사 예시 검색 중...", file=sys.stderr)
        examples = retrieve_similar(source_text[:500], args.user, k=args.k, type_filter=args.type_filter)
        _print_retrieved(examples)
        print("[cli] (2/2) 생성 모델 호출 중...", file=sys.stderr)
        print(transform_style(source_text, examples))


if __name__ == "__main__":
    main()
