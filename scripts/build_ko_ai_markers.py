"""한국어 'AI 과용 표현' 후보 사전 만들기 (영어 클리셰 사전 src/cliches.py의 한국어판, API 없음).

영어 사전은 사람이 고른 목록이라 근거가 약하다(D-100). 한국어는 손으로 고르지 않고 데이터에서 뽑는다:
같은 내용을 AI가 다시 쓴 글과 사람 원문을 짝지어(data/ko_pairs/raw.jsonl, results/ko_style/neutral.jsonl)
AI 쪽에 유독 많은 형태소·형태소 2개 묶음을 log-odds(사전 분포 포함)로 고른다. 같은 내용끼리 비교하므로
주제 차이가 아니라 말투 차이가 남는다. 그래도 주제어가 섞이지 않게 명사·숫자·외국어·일반 동사는 뺀다.
'-습니다'류 종결은 AI 습관이라기보다 말투(높임) 차이라 따로 'register'로 둔다.

결과: data/ko_ai_markers.json  (reviewed: false → 사람이 목록을 읽고 빼거나 확정할 것. 판단은 사람이 한다)
  python scripts/build_ko_ai_markers.py
"""
import json, math, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "ko_ai_markers.json"
EXCLUDE_TAG = ("NNG", "NNP", "NR", "SL", "SN", "SH", "VV", "VA", "XR", "NNB")
Z_MIN, AI_MIN, PAIR_MIN = 1.6, 8, 6


def load_pairs():
    pairs = []
    for p, ai_k, hu_k in ((ROOT / "data/ko_pairs/raw.jsonl", "ai", "original"), (ROOT / "results/ko_style/neutral.jsonl", "neutral", "source")):
        if p.exists():
            for l in p.open(encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    if r.get(ai_k) and r.get(hu_k): pairs.append((r[ai_k], r[hu_k]))
    return pairs


def units(text, kiwi):
    toks = [(t.form, t.tag) for t in kiwi.tokenize(text) if not t.tag.startswith("S")]
    u = [f"{f}/{g}" for f, g in toks]
    return u + [u[i] + " " + u[i + 1] for i in range(len(u) - 1)]


def keep(unit):
    return not any(part.split("/")[1].startswith(EXCLUDE_TAG) for part in unit.split(" "))


def label(unit):
    parts = [p.split("/") for p in unit.split(" ")]
    s = "".join(f for f, _ in parts)
    return ("-" if parts[0][1].startswith(("E", "J", "X", "VX", "VCP")) else "") + s


def main():
    from src.ko_style import _kiwi
    kiwi = _kiwi()
    pairs = load_pairs()
    if not pairs: sys.exit("[중단] 짝 데이터가 없습니다.")
    A, H, more, docs = Counter(), Counter(), Counter(), Counter()
    for a, h in pairs:
        ca, ch = Counter(units(a, kiwi)), Counter(units(h, kiwi)); A += ca; H += ch
        for w in set(ca) | set(ch):
            if ca[w] > ch[w]: more[w] += 1
            if ca[w]: docs[w] += 1
    na, nh, prior = sum(A.values()), sum(H.values()), A + H
    n0, a0 = sum(prior.values()), 500
    rows = []
    for w in prior:
        if not keep(w) or A[w] < AI_MIN or more[w] < PAIR_MIN or H[w] * 2 > A[w]: continue
        aw = prior[w] / n0 * a0
        d = math.log((A[w] + aw) / (na + a0 - A[w] - aw)) - math.log((H[w] + aw) / (nh + a0 - H[w] - aw))
        z = d / math.sqrt(1 / (A[w] + aw) + 1 / (H[w] + aw))
        if z < Z_MIN: continue
        kind = "register" if "니다/EF" in w else "ai_marker"
        rows.append({"unit": w, "label": label(w), "kind": kind, "z": round(z, 2), "ai": A[w], "human": H[w], "pairs_ai_more": more[w]})
    rows.sort(key=lambda r: -r["z"])
    # 겹침 정리(중복 계산 방지): ① 1개짜리가 거의 늘 어떤 2개 묶음 안에서만 나오면(묶음 빈도 ≥ 0.8배) 묶음만 남김
    # ② 남은 1개짜리를 포함하는 2개 묶음은 뺌
    bi = [r for r in rows if " " in r["unit"]]
    rows = [r for r in rows if " " in r["unit"] or not any(r["unit"] in b["unit"].split(" ") and b["ai"] >= 0.8 * r["ai"] for b in bi)]
    uni = {r["unit"] for r in rows if " " not in r["unit"]}
    rows = [r for r in rows if " " not in r["unit"] or not any(p in uni for p in r["unit"].split(" "))]
    OUT.write_text(json.dumps({"reviewed": False, "n_pairs": len(pairs),
                               "method": "AI 재작성 ↔ 같은 내용 사람 원문 짝, log-odds z ≥ %.1f, AI 빈도 ≥ %d, AI가 더 많은 짝 ≥ %d, 사람 빈도 ≤ AI/2, 명사·숫자·외국어·일반 동사/형용사 제외" % (Z_MIN, AI_MIN, PAIR_MIN),
                               "markers": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"짝 {len(pairs)}개 → 후보 {len(rows)}개 (AI 표현 {sum(r['kind'] == 'ai_marker' for r in rows)}, 말투 {sum(r['kind'] == 'register' for r in rows)}) → {OUT.relative_to(ROOT)}")
    for r in rows: print(f"  {r['z']:5.2f} {r['kind']:9s} {r['label']:10s} ({r['unit']})  AI {r['ai']} / 사람 {r['human']}")


if __name__ == "__main__":
    main()
