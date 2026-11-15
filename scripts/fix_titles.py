"""위키문헌이 동명 작품 구분용으로 붙인 제목 끝 괄호를 지운다. 예: "행복 (이상)" → "행복", "사랑 (수필)" → "사랑".
제목은 AI 입력 글의 주제로 쓰이므로, 괄호가 남으면 작가 이름이 주제에 섞인다.

쓰는 법 (프로젝트 루트): python scripts/fix_titles.py leegwangsu leeyuksa leesang
  → data/raw/corpus_<id>_raw.jsonl, data/raw/corpus_<id>_n*_raw.jsonl 의 title,
    data/examples/corpus_<id>*_examples.jsonl / *_train.jsonl 의 topic 을 고친다. 본문과 임베딩은 그대로.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def fix_file(path, key):
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    changed = []
    for r in rows:
        old = r.get(key, "")
        new = PAREN.sub("", old).strip()
        if new and new != old:
            r[key] = new
            changed.append((old, new))
    if changed:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return changed


def main(pids):
    if not pids:
        sys.exit("사용법: python scripts/fix_titles.py leegwangsu leeyuksa leesang")
    seen = set()
    for pid in pids:
        files = [(p, "title") for p in sorted((ROOT / "data" / "raw").glob(f"corpus_{pid}*_raw.jsonl"))]
        files += [(p, "topic") for p in sorted((ROOT / "data" / "examples").glob(f"corpus_{pid}*.jsonl"))]
        for path, key in files:
            for old, new in fix_file(path, key):
                if (pid, old) not in seen:
                    seen.add((pid, old))
                    print(f"  {pid}: {old} → {new}")
    print(f"고친 제목 {len(seen)}개")


if __name__ == "__main__":
    main(sys.argv[1:])
