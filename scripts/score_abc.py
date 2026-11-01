"""A/B/C 결과 채점 → results/abc/scores.csv(번역 1개당 1줄), results/abc/summary.csv(조건별 평균).

사용법 (프로젝트 루트): python scripts/score_abc.py
API를 부르지 않고 바로 끝납니다. scipy가 있으면 C vs A 짝 비교 검정도 출력합니다.

지표
- 의미: verdict(PASS 비율), meaning_score — A·B·C 모두 같은 검증기 v2로 잰 값
- AI 획일화: cliche_per100w (팀 사전 src/cliches.py, 100단어당), ttr, mtld
- 개인 지시 반영: 강희 since/because, 나경 however/therefore (100단어당), 수영 평균 문장 길이
- 기준선 '실제 영어 글': data/eval/english_<user>.jsonl을 같은 방식으로 잰 값.
  번역에는 전혀 쓰지 않고 채점 기준으로만 씀 → C가 A보다 이 값에 가까우면 '그 사람다운 영어'에 가까워진 것.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.cliches import ENGLISH_AI_CLICHES  # noqa: E402

OUT_DIR = ROOT / "results" / "abc"
RESULTS = OUT_DIR / "results.jsonl"

# 개인 지시로 쓰라고 한 단어는 상투어에서 제외 (지시를 잘 따를수록 감점되는 모순 방지)
INSTRUCTED = {"however", "therefore", "since", "because"}
CLICHES = sorted({c for group in ENGLISH_AI_CLICHES.values() for c in group} - INSTRUCTED,
                 key=len, reverse=True)
FEATURES = {"ganghee": ("since_because", ["since", "because"]),
            "nakyung": ("however_therefore", ["however", "therefore"])}
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


def words(t):
    return [w.lower() for w in WORD_RE.findall(t)]


def sentences(t):
    return [s for s in re.split(r"(?<!\b[A-Z]\.)(?<=[.!?])\s+", t.strip()) if WORD_RE.search(s)]


def _mtld_pass(ws, thr=0.72):
    factors, types, count = 0.0, set(), 0
    for w in ws:
        types.add(w)
        count += 1
        if len(types) / count <= thr:
            factors, types, count = factors + 1, set(), 0
    if count:
        factors += (1 - len(types) / count) / (1 - thr)
    return len(ws) / factors if factors else float(len(ws))


def mtld(ws):
    return (_mtld_pass(ws) + _mtld_pass(ws[::-1])) / 2 if ws else 0.0


def cliche_count(t):
    low, n = t.lower(), 0
    for c in CLICHES:  # 긴 표현부터 세고 지워서 'delve into'와 'delve'가 이중으로 세지지 않게
        hits = re.findall(r"\b" + re.escape(c) + r"\b", low)
        n += len(hits)
        low = re.sub(r"\b" + re.escape(c) + r"\b", " ", low)
    return n


def measure(text, person):
    ws, ss = words(text), sentences(text)
    per100 = lambda k: round(100 * k / len(ws), 2) if ws else 0.0  # noqa: E731
    row = {"n_words": len(ws), "avg_sent_len": round(len(ws) / len(ss), 2) if ss else 0.0,
           "ttr": round(len(set(ws)) / len(ws), 4) if ws else 0.0, "mtld": round(mtld(ws), 2),
           "cliche_per100w": per100(cliche_count(text))}
    for p, (name, targets) in FEATURES.items():
        row[f"{name}_per100w"] = per100(sum(ws.count(w) for w in targets)) if p == person else ""
    return row


METRICS = ["n_words", "avg_sent_len", "ttr", "mtld", "cliche_per100w",
           "since_because_per100w", "however_therefore_per100w", "meaning_score"]


def main():
    if not RESULTS.exists():
        sys.exit(f"{RESULTS.relative_to(ROOT)}이 없습니다. 먼저 python scripts/run_abc.py를 실행하세요.")
    latest = {}
    for line in RESULTS.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            key = (r["sample_id"], r["condition"])
            if r.get("output") or key not in latest:  # 같은 건이 여러 번이면 마지막 성공 결과
                latest[key] = r

    scored = []
    for r in sorted(latest.values(), key=lambda r: (r["sample_id"], r["condition"])):
        row = {"sample_id": r["sample_id"], "person": r["person"], "condition": r["condition"],
               "verdict": r.get("verdict", ""), "meaning_score": r.get("meaning_score", ""),
               "issues": r.get("issues", ""), "error": r.get("error", "")}
        row.update(measure(r.get("output") or "", r["person"]))
        scored.append(row)

    with (OUT_DIR / "scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)

    groups = defaultdict(list)
    for s in scored:
        if not s["error"]:
            groups[("전체", s["condition"])].append(s)
            groups[(s["person"], s["condition"])].append(s)

    summary = []
    for (person, cond), items in sorted(groups.items()):
        line = {"person": person, "condition": cond, "n": len(items)}
        for m in METRICS:
            vals = [float(i[m]) for i in items if i.get(m) not in ("", None)]
            line[m] = round(mean(vals), 3) if vals else ""
            if person == "전체" and m.endswith("_per100w") and m != "cliche_per100w":
                line[m] = ""  # 사람별 지시 지표는 그 사람 줄에서만 의미가 있음
        vs = [i["verdict"] for i in items if i["verdict"]]
        line["pass_rate"] = round(vs.count("PASS") / len(vs), 3) if vs else ""
        line["fail_rate"] = round(vs.count("FAIL") / len(vs), 3) if vs else ""
        summary.append(line)

    # 기준선: 그 사람의 실제 영어 글
    for person in sorted({s["person"] for s in scored}):
        path = ROOT / "data" / "eval" / f"english_{person}.jsonl"
        if not path.exists():
            continue
        docs = [json.loads(l)["text"] for l in path.open(encoding="utf-8") if l.strip()]
        ms = [measure(t, person) for t in docs]
        line = {"person": person, "condition": "실제 영어 글", "n": len(ms)}
        for m in METRICS:
            vals = [float(x[m]) for x in ms if x.get(m) not in ("", None)]
            line[m] = round(mean(vals), 3) if vals else ""
        line["pass_rate"] = line["fail_rate"] = ""
        summary.append(line)
    summary.sort(key=lambda l: (l["person"] != "전체", l["person"], l["condition"]))

    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    errors = sum(1 for s in scored if s["error"])
    print(f"scores.csv {len(scored)}줄, summary.csv {len(summary)}줄 → results/abc/")
    if errors:
        print(f"주의: 실패 {errors}건은 평균에서 뺐습니다. run_abc.py를 다시 돌리면 채워집니다.")
    show = ["person", "condition", "n", "pass_rate", "meaning_score", "cliche_per100w", "mtld", "avg_sent_len",
            "since_because_per100w", "however_therefore_per100w"]
    print("\n" + " | ".join(show))
    for l in summary:
        print(" | ".join(str(l.get(k, "")) for k in show))

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("\n(scipy가 없어 검정은 건너뜀: pip install scipy)")
        return
    by = {(s["sample_id"], s["condition"]): s for s in scored if not s["error"]}
    ids = sorted({s["sample_id"] for s in scored})
    print("\nC vs A, 같은 원문끼리 비교 (Wilcoxon)")
    for m in ["meaning_score", "cliche_per100w", "mtld", "avg_sent_len"]:
        pairs = [(float(by[(i, "C")][m]), float(by[(i, "A")][m])) for i in ids
                 if (i, "C") in by and (i, "A") in by and by[(i, "C")][m] != "" and by[(i, "A")][m] != ""]
        if len(pairs) < 6 or all(c == a for c, a in pairs):
            print(f"  {m}: 짝이 부족하거나 차이 없음 (n={len(pairs)})")
            continue
        _, p = wilcoxon([c for c, _ in pairs], [a for _, a in pairs])
        print(f"  {m}: 평균 차이(C-A) {mean(c - a for c, a in pairs):+.3f}, p={p:.3f}, n={len(pairs)}")


if __name__ == "__main__":
    main()
