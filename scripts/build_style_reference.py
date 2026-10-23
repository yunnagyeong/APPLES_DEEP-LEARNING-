"""사람별 문서 분포(기준 분포)를 계산해 data/style_reference.json으로 저장한다 (투두 23번).

generate.py가 스타일 지표를 "절대 임계값"이 아니라 효과 크기 d로 해석하도록 사람별 평균과
사람 안 표준편차를 제공한다: d = (본인 평균 − 나머지 사람들 평균) / 사람 안(풀링) 표준편차. 지표나 코퍼스가 바뀌면 다시 실행한다.

사용법 (프로젝트 루트에서): python scripts/build_style_reference.py [--save-docs]
- 모집단: data/raw/corpus_*_raw.jsonl 중 type이 essay/report인 문서 (stylometry_check.py와 같은 모집단).
  개념 노트·기획서는 설명용 글이라 제외.
- 문서별 값: 프로필과 같은 함수(src/style_features.py)로 문서 하나씩 계산. MTLD는 내용어 100개 이상
  문서만 사용(짧은 글은 불안정).
- 한계: 프로필은 예시 전체를 합쳐(문장 단위로 풀링) 계산하고 기준 분포는 문서별 값이라 계산 방식이
  완전히 같지는 않다. 작성자 수 3명·문서 67편 소표본이다.
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kiwipiepy  # noqa: E402

from src import style_features as sf  # noqa: E402

AUTHORS = ["ganghee", "nakyung", "suyoung"]
TYPES = {"essay", "report"}
# 프로필 키 이름 그대로 저장한다.
METRICS = ["sentence_length_avg", "directness_score", "logical_connective_rate", "lexical_diversity"]
MIN_CONTENT_TOKENS = 100  # MTLD 안정성 기준


def doc_metrics(text: str) -> dict:
    sents = sf.split_sentences(text)
    lem = [t.form for t in sf._get_kiwi().tokenize(" ".join(sents)) if t.tag in sf.CONTENT_TAGS]
    return {
        "n_sentences": len(sents),
        "n_content_tokens": len(lem),
        "sentence_length_avg": sf.sentence_length_avg(sents),
        "directness_score": sf.directness_score(sents),
        "logical_connective_rate": sf.logical_connective_rate(sents),
        "lexical_diversity": sf.calculate_mtld(lem) if len(lem) >= MIN_CONTENT_TOKENS else None,
    }


N_BOOT = 2000
SIGN_KEEP_MIN = 0.90  # generate.py가 지시문을 만들 때 요구하는 부호 유지 비율


def _effect_d(by: dict[str, list[float]], user: str) -> float:
    """d = (본인 문서 평균 - 나머지 사람들 문서 평균) / 사람 안 풀링 SD."""
    within = (sum((len(x) - 1) * st.variance(x) for x in by.values())
              / sum(len(x) - 1 for x in by.values())) ** 0.5
    others = [st.mean(x) for u, x in by.items() if u != user]
    return (st.mean(by[user]) - sum(others) / len(others)) / within


def sign_keep(rows: list[dict], m: str, seed: int = 0) -> dict[str, float]:
    """문서를 사람별로 다시 뽑아(부트스트랩 N_BOOT회) d의 부호가 원래와 같은 비율."""
    rng = random.Random(seed)
    docs = {u: [r[m] for r in rows if r["author"] == u and r[m] is not None] for u in AUTHORS}
    d0 = {u: _effect_d(docs, u) for u in AUTHORS}
    keep = {u: 0 for u in AUTHORS}
    for _ in range(N_BOOT):
        b = {u: [rng.choice(v) for _ in v] for u, v in docs.items()}
        for u in AUTHORS:
            if (_effect_d(b, u) > 0) == (d0[u] > 0):
                keep[u] += 1
    return {u: round(keep[u] / N_BOOT, 3) for u in AUTHORS}


def main() -> None:
    rows = []
    for u in AUTHORS:
        for line in open(ROOT / f"data/raw/corpus_{u}_raw.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("type") not in TYPES:
                continue
            rows.append({"author": u, "id": d["id"], "type": d["type"], **doc_metrics(d["text"])})

    ref = {}
    for m in METRICS:
        v = [r[m] for r in rows if r[m] is not None]
        by = {u: [r[m] for r in rows if r["author"] == u and r[m] is not None] for u in AUTHORS}
        # 사람 안 표준편차: 사람별 분산을 자유도로 가중해 합친 풀링 SD
        within = (sum((len(x) - 1) * st.variance(x) for x in by.values())
                  / sum(len(x) - 1 for x in by.values())) ** 0.5
        ref[m] = {
            "mean": round(st.mean(v), 4), "sd": round(st.stdev(v), 4), "n": len(v),
            "within_sd": round(within, 4),
            "author_mean": {u: round(st.mean(x), 4) for u, x in by.items()},
            "author_sd": {u: round(st.stdev(x), 4) for u, x in by.items()},
            "author_n": {u: len(x) for u, x in by.items()},
            "author_d": {u: round(_effect_d(by, u), 3) for u in AUTHORS},
            "author_sign_keep": sign_keep(rows, m),
        }
    out = {
        "population": "raw essay+report documents of ganghee/nakyung/suyoung",
        "n_docs": len(rows),
        "per_author_docs": {u: sum(1 for r in rows if r["author"] == u) for u in AUTHORS},
        "kiwipiepy": kiwipiepy.__version__,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": ref,
    }
    path = ROOT / "data" / "style_reference.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장: {path}  (문서 {len(rows)}편, kiwipiepy {kiwipiepy.__version__})")
    for m in METRICS:
        print(f"  {m:25s} d {ref[m]['author_d']} 부호유지 {ref[m]['author_sign_keep']}\n  {'':25s} mean {ref[m]['mean']:8.3f}  sd {ref[m]['sd']:7.3f}  사람안 sd {ref[m]['within_sd']:7.3f}  n={ref[m]['n']}")

    if "--save-docs" in sys.argv:
        p = ROOT / "data" / "measure" / f"style_reference_docs_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"문서별 값 저장: {p}")


if __name__ == "__main__":
    main()
