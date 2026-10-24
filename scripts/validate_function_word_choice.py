"""기능어 선택 프로필(24번) 검증: 이 지표 묶음만으로 작성자를 얼마나 맞히는가 (stylometry_check.py와 같은 방법).

사용법: python scripts/validate_function_word_choice.py [--save]
비교 대상(기준선, 26번): 기능 형태소 top100 → 작성자 LOO 0.90 / 교차 장르 0.76 / 장르 맞히기 0.69
계산: 문서별 그룹 빈도(형태소 1000개당) → Burrows' Delta 최근접 중심(leave-one-out), 교차 장르(다른 장르 문서로만
학습), 장르 맞히기. LLM·API 없음.
주의: 그룹 목록은 D-003이 미리 정한 묶음(대조·결론·이유·목적)을 따랐지만, 그 안의 표현 선택과 영어 매핑은
같은 코퍼스의 d를 보면서 골랐다. 그래서 여기 나오는 정확도는 선택 편향이 있어 낙관적일 수 있다.
"""
import json, sys, time
from collections import namedtuple
from pathlib import Path

import numpy as np
from kiwipiepy import Kiwi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import style_features as sf  # noqa: E402
from stylometry_check import AUTHORS, FUNC, delta_loo, matrix  # noqa: E402

Tok = namedtuple("Tok", "form tag")


def load_docs():
    kiwi = Kiwi()
    docs = []
    for who in AUTHORS:
        for line in open(ROOT / f"data/raw/corpus_{who}_raw.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("type") not in ("essay", "report"):
                continue
            toks = [(t.form, t.tag) for t in kiwi.tokenize(o["text"])]
            docs.append(dict(author=who, type=o["type"], toks=toks))
    return docs


def acc(X, A, G):
    idx = np.arange(len(A))
    return dict(
        author_loo=round(float(np.mean(delta_loo(X, A) == A)), 2),
        cross_genre=round(float(np.mean(delta_loo(X, A, lambda i: (idx != i) & (G != G[i])) == A)), 2),
        genre_guess=round(float(np.mean(delta_loo(X, G) == G)), 2),
    )


def main():
    docs = load_docs()
    A = np.array([d["author"] for d in docs]); G = np.array([d["type"] for d in docs])
    groups = list(sf.FUNCTION_WORD_GROUPS)
    Xg = np.array([[1000 * c / len(d["toks"]) for c in
                    (sf.count_function_word_groups([Tok(*t) for t in d["toks"]])[g] for g in groups)] for d in docs])
    Xf, _ = matrix(docs, FUNC, 100)
    rng = np.random.default_rng(0)
    out = {"기능어 선택 그룹 %d개" % len(groups): acc(Xg, A, G),
           "기능 형태소 top100 (기준선)": acc(Xf, A, G),
           "top100 + 그룹": acc(np.hstack([Xf, Xg]), A, G)}
    # 비교용: 기능 형태소 중 무작위 %d개 (같은 개수의 특징으로 얻는 정확도, 50회 평균)
    k = len(groups)
    rs = [acc(Xf[:, rng.choice(Xf.shape[1], k, replace=False)], A, G) for _ in range(50)]
    out[f"기능 형태소 중 무작위 {k}개 (50회 평균)"] = {m: round(float(np.mean([r[m] for r in rs])), 2) for m in rs[0]}
    print(f"문서 {len(docs)}편")
    for name, r in out.items():
        print(f"[{name}] 작성자 LOO {r['author_loo']:.2f} (우연 0.33) | 교차 장르 {r['cross_genre']:.2f} | 장르 맞히기 {r['genre_guess']:.2f}")
    if "--save" in sys.argv:
        p = ROOT / "data" / "measure" / f"fwc_validation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        json.dump(dict(n_docs=len(docs), groups=groups, result=out), open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("저장:", p)


if __name__ == "__main__":
    main()
