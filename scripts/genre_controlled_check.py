"""3번 남은 것: 장르를 통제한 개인차 확인 (type=essay/report). LLM·API 없음.

질문: 사람 간 차이가 장르 차이(에세이 vs 보고서)로 설명되는가, 같은 장르 안에서도 남는가?
1) 문서별 지표(data/measure/style_reference_docs_*.csv, `build_style_reference.py --save-docs`):
   - 장르 평균을 뺀 값(장르 통제)으로 작성자 η² 와 층화 순열 p(작성자 라벨을 같은 장르 안에서만 섞음)
   - 장르별(에세이만 / 보고서만) η² 와 순열 p. 작성자와 장르가 겹치는 정도(강희 에세이 3편)에 주의
2) 같은 장르 안에서만 작성자 맞히기(기능 형태소 top100, 기능어 6그룹): 학습·평가를 한 장르 안에서만(LOO),
   다수결(가장 많은 작성자 찍기) 기준선과 함께 표기.
사용법: python scripts/genre_controlled_check.py [docs.csv] [--save]
"""
import csv, glob, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import stylometry_check as sc  # noqa: E402

METRICS = ["sentence_length_avg", "directness_score", "logical_connective_rate", "lexical_diversity",
           "fw_however", "fw_but", "fw_whereas", "fw_therefore", "fw_causal", "fw_purpose_formal"]
N_PERM = 2000
rng = np.random.default_rng(0)


def eta2(v, a):
    gm = v.mean()
    ss_t = ((v - gm) ** 2).sum()
    if ss_t == 0:
        return 0.0
    ss_b = sum(((a == k).sum()) * (v[a == k].mean() - gm) ** 2 for k in set(a))
    return ss_b / ss_t


def strat_perm(v, a, g, stat):
    obs = stat(v, a)
    cnt = 0
    for _ in range(N_PERM):
        a2 = a.copy()
        for k in set(g):
            m = np.where(g == k)[0]
            a2[m] = rng.permutation(a[m])
        cnt += stat(v, a2) >= obs
    return obs, (cnt + 1) / (N_PERM + 1)


def load_rows(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return [r for r in rows if r["type"] in ("essay", "report")]


def metric_table(rows):
    A = np.array([r["author"] for r in rows]); G = np.array([r["type"] for r in rows])
    out = {}
    for m in METRICS:
        ok = np.array([r[m] not in ("", None) for r in rows])
        v = np.array([float(r[m]) if r[m] not in ("", None) else np.nan for r in rows])
        a, g, x = A[ok], G[ok], v[ok]
        res = x.copy()
        for k in set(g):
            res[g == k] -= x[g == k].mean()
        raw_e = eta2(x, a)
        ctrl_e, ctrl_p = strat_perm(res, a, g, eta2)
        per = {}
        for k in ("essay", "report"):
            mk = g == k
            e, p = strat_perm(x[mk], a[mk], np.zeros(mk.sum(), dtype=int), eta2)
            per[k] = (round(e, 2), round(p, 3))
        gen_e = eta2(x, g)  # 장르가 설명하는 비율
        out[m] = dict(n=int(ok.sum()), eta2_raw=round(raw_e, 2), eta2_genre_ctrl=round(ctrl_e, 2),
                      p_strat=round(ctrl_p, 3), eta2_genre=round(gen_e, 2), essay=per["essay"], report=per["report"])
    return out


def within_genre_acc(docs):
    """한 장르 안에서만 학습·평가(LOO). 기능 형태소 top100, 기능어 6그룹."""
    res = {}
    for gk in ("essay", "report"):
        ds = [d for d in docs if d["type"] == gk]
        A = np.array([d["author"] for d in ds])
        base = max(Counter(A).values()) / len(A)
        Xf, _ = sc.matrix(ds, sc.FUNC, 100)
        pf = sc.delta_loo(Xf, A)
        res[gk] = dict(n=len(ds), counts=dict(Counter(A)), majority=round(base, 2),
                       func_top100=round(float(np.mean(pf == A)), 2))
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else sorted(glob.glob(str(ROOT / "data/measure/style_reference_docs_*.csv")))[-1]
    rows = load_rows(path)
    A = [r["author"] for r in rows]; G = [r["type"] for r in rows]
    print("문서", len(rows), {f"{a}/{g}": c for (a, g), c in sorted(Counter(zip(A, G)).items())}, "|", Path(path).name)
    mt = metric_table(rows)
    print("\n지표별 작성자 η² (원값 / 장르 통제 + 층화 순열 p) | 장르가 설명하는 η² | 장르 안: 에세이 η²(p) 보고서 η²(p)")
    for m, r in mt.items():
        print(f"  {m:22s} n={r['n']:2d}  {r['eta2_raw']:.2f} / {r['eta2_genre_ctrl']:.2f} (p={r['p_strat']:.3f}) | 장르 {r['eta2_genre']:.2f} | "
              f"에세이 {r['essay'][0]:.2f}({r['essay'][1]:.3f}) 보고서 {r['report'][0]:.2f}({r['report'][1]:.3f})")
    docs = sc.load(str(ROOT / "data" / "raw"))
    wg = within_genre_acc(docs)
    print("\n같은 장르 안에서만 작성자 맞히기 (기능 형태소 top100, LOO)")
    for gk, r in wg.items():
        print(f"  {gk}: n={r['n']} {r['counts']} | 다수결 기준선 {r['majority']:.2f} | top100 {r['func_top100']:.2f}")
    if "--save" in sys.argv:
        out = ROOT / "data" / "measure" / f"genre_controlled_{time.strftime('%Y%m%d_%H%M%S')}.json"
        json.dump(dict(source=Path(path).name, metrics=mt, within_genre=wg), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("저장:", out)
