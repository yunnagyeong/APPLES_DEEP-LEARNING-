"""지시문 수정(22번) 전/후 프롬프트로 같은 문장을 번역해 나란히 비교하는 스크립트.

- 번역 생성만 호출합니다(claim 추출·NLI 없음). claim 모델의 하루 20회 한도를 쓰지 않고,
  호출은 케이스당 임베딩 1회(검색) + 생성 2×n회입니다.
- "수정 전" 프롬프트는 git HEAD의 src/generate.py를, "수정 후"는 작업 폴더의 파일을 씁니다.
  (HEAD가 아직 수정 전 버전일 때만 의미가 있습니다. 수정 사항을 커밋했다면 --base <커밋>을 주세요.)
- 같은 입력에는 같은 RAG 예시를 양쪽에 넣어 지시문 차이만 보이게 합니다.
- 결과 해석은 사람이 합니다("AI가 AI를 평가하지 않는다"). 영어 헤징 단어 개수는 눈으로 훑을 때 쓰는 참고용일 뿐입니다.

사용법:
  python scripts/compare_prompt_change.py --dry                  # API 호출 없이 프롬프트 차이만 출력
  python scripts/compare_prompt_change.py --user ganghee --ids 1,2,3,4,5,6,7,12,28 --n 2
결과: data/measure/prompt_compare_<user>_<시각>.csv
"""

import argparse
import csv
import importlib.util
import re
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HEDGE_RE = re.compile(
    r"\b(may|might|could|can|perhaps|possibly|probably|seems?|seemed|appears?|"
    r"i think|i believe|not sure|not certain|likely|somewhat|tend|suggest)\b",
    re.IGNORECASE,
)


def load_base_generate(ref: str):
    """git <ref>의 src/generate.py를 별도 모듈로 읽어옵니다."""
    src = subprocess.run(
        ["git", "show", f"{ref}:src/generate.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    mod = types.ModuleType("src.generate_base")
    mod.__package__ = "src"
    mod.__file__ = str(ROOT / "src" / "generate_base.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


class _FakeClient:
    """--dry용: 프롬프트만 기록하고 API는 부르지 않습니다."""

    def __init__(self, sink):
        self.sink = sink
        self.models = self

    def generate_content(self, model, contents):
        self.sink.append(contents)
        return types.SimpleNamespace(text="[dry-run: 번역 생략]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="ganghee")
    ap.add_argument("--ids", default="1,2,3,4,5,6,7,12,28", help="error_cases.csv 번호")
    ap.add_argument("--n", type=int, default=2, help="조건당 샘플 수(생성은 매번 달라짐)")
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    import json

    from src import config
    from src import generate as new

    old = load_base_generate(args.base)
    profile = json.load(open(config.style_profile_cache_path(args.user), encoding="utf-8"))

    cases = {r["번호"]: r for r in csv.DictReader(open(ROOT / "error_cases.csv", encoding="utf-8-sig"))}
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]

    print("=== 지시문 (수정 전) ===\n" + old._style_profile_to_instructions(profile))
    print("\n=== 지시문 (수정 후) ===\n" + new._style_profile_to_instructions(profile))

    prompts = []
    if args.dry:
        old._ensure_client = lambda: _FakeClient(prompts)
        new._ensure_client = lambda: _FakeClient(prompts)
        examples = []
    from src.retrieve import retrieve_similar

    rows = []
    for cid in ids:
        ko = cases[cid]["원문"]
        if not args.dry:
            examples = retrieve_similar(ko, args.user, k=config.DEFAULT_TOP_K)
        for label, mod in (("수정 전", old), ("수정 후", new)):
            for k in range(1, args.n + 1):
                en = mod.translate_with_style(ko, profile, args.user, examples=examples)
                rows.append(
                    {
                        "번호": cid, "원문": ko, "조건": label, "샘플": k, "번역": en.strip(),
                        "영어 헤징 단어 수(참고)": len(HEDGE_RE.findall(en)),
                        "기존 사람 라벨": cases[cid]["문제유형"],
                    }
                )
        if not args.dry:
            time.sleep(1)

    if args.dry:
        a, b = prompts[0], prompts[args.n]
        print("\n=== 첫 케이스 프롬프트 차이 (줄 단위) ===")
        import difflib

        for line in difflib.unified_diff(a.splitlines(), b.splitlines(), "수정 전", "수정 후", lineterm="", n=0):
            print(line)
        return

    out = ROOT / "data" / "measure" / f"prompt_compare_{args.user}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for cid in ids:
        print(f"\n[{cid}] {cases[cid]['원문']}")
        for r in rows:
            if r["번호"] == cid:
                print(f"  ({r['조건']} #{r['샘플']}, 헤징어 {r['영어 헤징 단어 수(참고)']}) {r['번역']}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
