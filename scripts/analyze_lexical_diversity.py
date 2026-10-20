"""사람별·문서별 MTLD(lexical_diversity) 분포를 계산해 generate.py의 LEXICAL_DIVERSITY_LOW/HIGH 기준값을 다시 정하는 스크립트.

사용법 (프로젝트 루트에서): python scripts/analyze_lexical_diversity.py
data/raw/corpus_*_raw.jsonl 전체를 읽는다. 문서별 값은 내용어 100개 이상인 문서만 사용(짧은 글은 MTLD가 불안정).
"""
import json, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import style_features as sf

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def lemmas(sentences):
    kiwi = sf._get_kiwi()
    toks = kiwi.tokenize(" ".join(sentences))
    return [t.form for t in toks if t.tag in sf.CONTENT_TAGS]

def q(v, p):
    v = sorted(v); k = (len(v) - 1) * p; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

people = {n: load(ROOT / f"data/raw/corpus_{n}_raw.jsonl") for n in ("ganghee", "suyoung", "nakyung")}
allperdoc = []
print("== 사람별 합본(프로필과 같은 방식) ==")
for n, docs in people.items():
    sents = []
    for d in docs: sents += sf.split_sentences(d["text"])
    lem = lemmas(sents)
    ttr = len(set(lem)) / len(lem)
    print(f"{n:8s} 문서 {len(docs):2d}편  내용어 {len(lem):6d}개  TTR {ttr:.3f}  MTLD {sf.calculate_mtld(lem):6.1f}")
print("\n== 문서별 MTLD (내용어 100개 이상인 문서만; 짧은 글은 MTLD가 불안정) ==")
for n, docs in people.items():
    vals = []
    for d in docs:
        lem = lemmas(sf.split_sentences(d["text"]))
        if len(lem) >= 100:
            vals.append(sf.calculate_mtld(lem))
    allperdoc += vals
    if vals:
        print(f"{n:8s} n={len(vals):2d} 중앙값 {st.median(vals):6.1f}  최소 {min(vals):6.1f}  최대 {max(vals):6.1f}")
    else:
        print(f"{n:8s} 100개 이상 문서 없음")
print(f"\n전체 문서 n={len(allperdoc)}  하위 1/3 경계 {q(allperdoc,1/3):.1f}  상위 1/3 경계 {q(allperdoc,2/3):.1f}  중앙값 {q(allperdoc,.5):.1f}  min {min(allperdoc):.1f} max {max(allperdoc):.1f}")
# 현재 프로필 방식(ganghee examples 177개 합본)
ex = load(ROOT / "data/examples/corpus_ganghee_examples.jsonl")
sents = []
for e in ex: sents += sf.split_sentences(e["text"])
lem = lemmas(sents)
print(f"\nganghee examples 합본(현재 프로필 입력): 내용어 {len(lem)}개 TTR {len(set(lem))/len(lem):.3f} MTLD {sf.calculate_mtld(lem):.1f}")
