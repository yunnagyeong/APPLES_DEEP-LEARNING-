"""한국어에서 잰 기능어 선택 방향이 각자의 영어 글에서도 같은 방향인지 확인 (확인용, 지시문 수정 근거로 쓰지 않음).
data/eval은 7주차 정답이므로 이 결과로 지시문을 다시 고치지 않는다(D-008 ⑦). API·LLM 없음.
사용법: python scripts/check_english_direction.py [--save]
"""
import json, re, sys, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
USERS = {"ganghee": "강희", "nakyung": "나경", "suyoung": "수영"}
PAT = {
    "however": r"\bhowever\b",
    "but": r"\bbut\b",
    "whereas": r"\bwhereas\b",
    "therefore": r"\b(?:therefore|thus|hence|consequently)\b",
    "causal(since/because)": r"\b(?:since|because|due to|owing to)\b",
    "purpose(in order to)": r"\b(?:in order to|so as to)\b",
}
# 한국어 d (B-012 표, + 이면 그 사람이 팀 나머지보다 많이 씀)
KO_D = {
    "however": {"ganghee": -1.16, "nakyung": 2.71, "suyoung": -1.55},
    "but": {"ganghee": 0.00, "nakyung": -0.54, "suyoung": 0.55},
    "whereas": {"ganghee": -0.30, "nakyung": 0.54, "suyoung": -0.24},
    "therefore": {"ganghee": -0.61, "nakyung": 0.86, "suyoung": -0.25},
    "causal(since/because)": {"ganghee": 1.31, "nakyung": -0.63, "suyoung": -0.68},
    "purpose(in order to)": {"ganghee": -0.82, "nakyung": -0.70, "suyoung": 1.52},
}
def load(u):
    return [json.loads(l) for l in open(ROOT / f"data/eval/english_{u}.jsonl", encoding="utf-8")]
def main():
    res = {}
    for u in USERS:
        docs = load(u)
        per = {k: [] for k in PAT}
        words = []
        for d in docs:
            t = d["text"].lower(); n = len(re.findall(r"[a-z']+", t)); words.append(n)
            for k, p in PAT.items():
                per[k].append(len(re.findall(p, t)) / max(n, 1) * 1000)
        res[u] = {"docs": len(docs), "words": sum(words), "per": per}
    print("영어 글: " + ", ".join(f"{USERS[u]} {r['docs']}편·{r['words']}단어" for u, r in res.items()))
    print("\n문서별 평균(1000단어당) [문서 중앙값, 등장 문서 수] / 한국어 d / 방향 일치(|d|<0.5는 -)")
    out = {}
    for k in PAT:
        print(f"\n{k}")
        means = {u: st.mean(res[u]["per"][k]) for u in USERS}
        for u in USERS:
            v = res[u]["per"][k]
            others = st.mean([means[o] for o in USERS if o != u])
            en_dir = means[u] - others
            ko = KO_D[k][u]
            same = "-" if abs(ko) < 0.5 else ("O" if (en_dir > 0) == (ko > 0) else "X")
            print(f"  {USERS[u]}: {means[u]:.2f} [중앙 {st.median(v):.2f}, {sum(x>0 for x in v)}/{len(v)}편]  나머지 평균 {others:.2f}  한국어 d {ko:+.2f}  방향 일치 {same}")
            out.setdefault(k, {})[u] = {"mean": means[u], "median": st.median(v), "nonzero_docs": sum(x > 0 for x in v), "n_docs": len(v), "others_mean": others, "ko_d": ko, "same_direction": same}
    if "--save" in sys.argv:
        p = ROOT / "data/measure/english_direction_check.json"
        p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"); print("\n저장:", p)
if __name__ == "__main__":
    main()
