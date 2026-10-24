"""5번: 기능어 그룹(FUNCTION_WORD_GROUPS) 정밀도 점검 시트 생성·채점. LLM·API 없음.

팀원 2명이 각자 "이 문장에서 [[표시]]된 표현이 그룹의 의도한 기능(대조·결론·이유·목적)으로 쓰였는가"를
O/X로 표시하면, 그룹별 정밀도(%)와 두 사람의 일치도(일치율, Cohen's kappa)가 나온다.
판단은 사람이 한다(AI가 리스트의 타당성을 평가하지 않는다).

만들기:  python scripts/make_wordlist_check_sheet.py            # 그룹당 20문장, 작성자별 균등 추출, seed 고정
  → data/measure/wordlist_check_sheet.csv (점검자에게 줌, 작성자·문서 정보 없음)
  → data/measure/wordlist_check_key.csv   (문장 id → 작성자·문서, 점검 끝난 뒤에 열어 볼 것)
채점:    python scripts/make_wordlist_check_sheet.py --score <채워진 시트.csv>
시트 열: 문장id, 그룹, 문장([[...]] 표시), 의도한 기능, 점검자A(O/X), 점검자B(O/X), 메모
"""
import csv, json, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import style_features as sf  # noqa: E402

AUTHORS = ["ganghee", "nakyung", "suyoung"]
PER_GROUP = 20
FUNC_DESC = {"however": "대조(격식, however)", "but": "대조(평이, but)", "whereas": "두 대상 대조(whereas)",
             "therefore": "결론(therefore)", "causal": "이유·원인(since/because)", "purpose_formal": "목적(in order to)"}


def mark(sent, toks, g):
    spans = [(t.start, t.start + t.len) for t in toks
             if any(t.form == f and t.tag.startswith(tag) for f, tag in sf.FUNCTION_WORD_GROUPS[g])]
    for s, e in sorted(spans, reverse=True):
        sent = sent[:s] + "[[" + sent[s:e] + "]]" + sent[e:]
    return sent, bool(spans)


def build():
    rng = random.Random(0)
    kiwi = sf._get_kiwi()
    pool = defaultdict(lambda: defaultdict(list))  # group -> author -> [(text, doc)]
    for a in AUTHORS:
        for line in open(ROOT / f"data/raw/corpus_{a}_raw.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("type") not in ("essay", "report"):
                continue
            for s in sf.split_sentences(d["text"]):
                toks = kiwi.tokenize(s)
                for g in sf.FUNCTION_WORD_GROUPS:
                    m, ok = mark(s, toks, g)
                    if ok:
                        pool[g][a].append((m, d["id"]))
    sheet, key, n = [], [], 0
    for g in sf.FUNCTION_WORD_GROUPS:
        for a in AUTHORS:
            rng.shuffle(pool[g][a])
        picked, i = [], 0
        while len(picked) < PER_GROUP and any(pool[g][a] for a in AUTHORS):  # 작성자 라운드로빈
            a = AUTHORS[i % 3]; i += 1
            if pool[g][a]:
                text, doc = pool[g][a].pop()
                picked.append((a, doc, text))
        rng.shuffle(picked)
        for a, doc, text in picked:
            n += 1
            sid = f"S{n:03d}"
            sheet.append({"문장id": sid, "그룹": g, "문장": text, "의도한 기능": FUNC_DESC[g], "점검자A": "", "점검자B": "", "메모": ""})
            key.append({"문장id": sid, "작성자": a, "문서": doc})
    out = ROOT / "data" / "measure"
    for name, rows in (("wordlist_check_sheet.csv", sheet), ("wordlist_check_key.csv", key)):
        with open(out / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cnt = defaultdict(int)
    for r in sheet:
        cnt[r["그룹"]] += 1
    print("그룹별 문장 수", dict(cnt), "→", out / "wordlist_check_sheet.csv")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in ("O", "X"))
    return None if pe == 1 else (po - pe) / (1 - pe)


def score(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    byg = defaultdict(list)
    for r in rows:
        a, b = r["점검자A"].strip().upper(), r["점검자B"].strip().upper()
        if a in ("O", "X") and b in ("O", "X"):
            byg[r["그룹"]].append((a, b))
    print(f"{'그룹':16s} {'n':>3s} {'A 정밀도':>8s} {'B 정밀도':>8s} {'일치율':>6s} {'kappa':>6s}")
    allp = []
    for g, ps in byg.items():
        A = [p[0] for p in ps]; B = [p[1] for p in ps]
        k = kappa(A, B); allp += ps
        print(f"{g:16s} {len(ps):3d} {A.count('O')/len(ps):8.0%} {B.count('O')/len(ps):8.0%} "
              f"{sum(x == y for x, y in ps)/len(ps):6.0%} {'-' if k is None else f'{k:.2f}':>6s}")
    if allp:
        print(f"{'전체':16s} {len(allp):3d} (n이 작아 구간이 넓음. 소표본·경향으로만 표기)")


if __name__ == "__main__":
    if "--score" in sys.argv:
        score(sys.argv[sys.argv.index("--score") + 1])
    else:
        build()
