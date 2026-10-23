"""개인차 진단: 규칙·통계만으로 '누가 쓴 글인지' 맞힐 수 있는지 확인 (D-003, 투두 26).

LLM 판단 없음. Kiwi 형태소 분석 + Burrows' Delta(z-점수 + 맨해튼 거리, 최근접 중심) leave-one-out.
- 기능 형태소(조사·어미·접속부사 등)로 작성자를 맞히면 → 데이터에 개인 문체가 있다는 근거
- 내용어(명사·동사)로만 맞히는 것보다 높고, 다른 장르로 학습해도 맞히면 → 주제·장르보다 문체 신호
- 스타일 지표를 바꿀 때마다 실행해 "개인 지표"의 기준(작성자 LOO·교차 장르 정확도 유지)을 확인한다.

사용법 (저장소 루트):
  python scripts/stylometry_check.py              # data/raw 사용
  python scripts/stylometry_check.py data/raw --save   # data/measure/stylometry_<시각>.json 저장

주의: raw의 `type`(essay/report)이 있는 문서만 사용한다(노트·기획서는 장르 혼입이라 제외).
Kiwi 버전에 따라 값이 조금 달라질 수 있어 결과에 버전을 같이 기록한다.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from kiwipiepy import Kiwi

ROOT = Path(__file__).resolve().parent.parent
AUTHORS = ["ganghee", "nakyung", "suyoung"]
FUNC = ("J", "E", "XS", "VX", "VCP", "VCN", "NNB", "MAJ", "XP")
CONTENT = ("NNG", "NNP", "VV", "VA")


def load(raw_dir):
    kiwi = Kiwi()
    docs = []
    for who in AUTHORS:
        with open(Path(raw_dir) / f"corpus_{who}_raw.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                toks = [(t.form, t.tag) for t in kiwi.tokenize(o["text"])]
                docs.append(dict(author=who, type=o.get("type"), toks=toks))
    return [d for d in docs if d["type"] in ("essay", "report")]


def matrix(docs, tags, n):
    tl = [[f"{f}/{t}" for f, t in d["toks"] if t.startswith(tags)] for d in docs]
    vocab = [w for w, _ in Counter(x for t in tl for x in t).most_common(n)]
    return np.array([[Counter(t)[w] / max(1, len(t)) for w in vocab] for t in tl]), vocab


def delta_loo(X, y, train_mask=None):
    idx = np.arange(len(y))
    preds = []
    for i in idx:
        tr = (idx != i) if train_mask is None else train_mask(i)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        cents = {a: Z[tr & (y == a)].mean(0) for a in set(y[tr])}
        preds.append(min(cents, key=lambda a: np.abs(Z[i] - cents[a]).mean()))
    return np.array(preds)


def run(docs):
    """문서 리스트로 3가지 정확도를 계산해 dict로 반환한다(지표 묶음을 바꿔 비교할 때 재사용)."""
    A = np.array([d["author"] for d in docs])
    G = np.array([d["type"] for d in docs])
    idx = np.arange(len(A))
    out = {}
    for name, tags, n in [("기능 형태소 top100", FUNC, 100), ("내용어 top200 (주제 대조군)", CONTENT, 200)]:
        X, _ = matrix(docs, tags, n)
        p = delta_loo(X, A)
        pc = delta_loo(X, A, lambda i: (idx != i) & (G != G[i]))
        pg = delta_loo(X, G)
        out[name] = dict(
            author_loo=round(float(np.mean(p == A)), 2),
            cross_genre=round(float(np.mean(pc == A)), 2),
            genre_guess=round(float(np.mean(pg == G)), 2),
        )
    return out, A, G


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw_dir = args[0] if args else str(ROOT / "data" / "raw")
    docs = load(raw_dir)
    result, A, G = run(docs)
    counts = {f"{a}/{g}": c for (a, g), c in sorted(Counter(zip(A, G)).items())}
    import kiwipiepy

    print(f"문서 {len(docs)}편 (에세이+보고서)", counts, f"| kiwipiepy {kiwipiepy.__version__}")
    for name, r in result.items():
        print(f"[{name}] 작성자 LOO {r['author_loo']:.2f} (우연 0.33) | 교차 장르 {r['cross_genre']:.2f} | 장르 맞히기 {r['genre_guess']:.2f}")
    if "--save" in sys.argv:
        out = ROOT / "data" / "measure" / f"stylometry_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(n_docs=len(docs), counts=counts, kiwipiepy=kiwipiepy.__version__, result=result),
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("저장:", out)
