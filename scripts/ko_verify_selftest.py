"""한→한 경량 검증기(src/ko_verify.py) 자체 시험. API 없음, 수 초.

① 오탐(정상인데 걸림): 의미가 같다고 보는 실제 쌍에서 PASS가 아닌 비율
   - 중립문↔사람 원문 12쌍 × 양방향(results/ko_style/neutral.jsonl): 문체를 크게 바꾼 쌍
     ※ 중립화 모델이 실제로 의미를 바꾼 쌍도 섞여 있다(예: nakyung_doc_009_03 '수 있다'→단정). 그래서 오탐 상한에 가깝다.
   - 기존 문체 변환 결과 18건(results/ko_doc/base, 입력 data/ko_docs/select): 문체를 조금만 바꾼 긴 글
② 탐지율(오류를 넣었을 때 잡힘): 위 정상 쌍의 출력에 오류 하나를 규칙으로 넣고, 넣은 유형이 잡히는지.
   두 묶음을 따로 본다: 중립문↔원문은 서로 많이 바꿔 쓴 글이라 서술어가 달라 어렵고(어려운 쪽),
   기존 변환 결과는 조금만 바꾼 글이다(쉬운 쪽). 실제 변환은 그 사이 어딘가.
   부정 뒤집기 / 숫자 바꾸기 / 문장 빼기 / 다른 글 문장 넣기 / 예시 문장 베끼기 / 추측→단정 / 범위어 넣기 / 고유명사 바꾸기 / 머리말

주의: 규칙으로 넣은 오류는 LLM이 실제로 내는 미묘한 오류보다 찾기 쉽다. 이 시험은 "이런 유형은 확실히 잡는다"를
      보여 주는 하한 점검이지 실제 성능 추정이 아니다. 실제 변환 결과는 사람이 일부 표본을 확인해 오탐·누락을 센다.

사용법: python scripts/ko_verify_selftest.py [--show]   (--show: 걸린 정상 쌍의 사유 출력)
"""
import argparse, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import ko_verify as kv  # noqa: E402

SEED = 7


def legit_pairs():
    pairs = []
    for l in (ROOT / "results/ko_style/neutral.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("neutral"):
            pairs.append({"id": f"{r['nid']}|AI→사람", "set": "중립문↔원문", "src": r["neutral"], "out": r["source"], "person": r["person"]})
            pairs.append({"id": f"{r['nid']}|사람→AI", "set": "중립문↔원문", "src": r["source"], "out": r["neutral"], "person": r["person"]})
    gen = ROOT / "results/ko_doc/base/gen.jsonl"
    if gen.exists():
        for l in gen.open(encoding="utf-8"):
            r = json.loads(l)
            if r.get("output") and not r.get("error"):
                src = (ROOT / "data/ko_docs" / r["set"] / f"{r['doc']}.txt").read_text(encoding="utf-8").strip()
                pairs.append({"id": f"{r['doc']}|{r['target']}|{r['cond']}", "set": "기존 변환 결과", "src": src,
                              "out": r["output"], "person": r["target"]})
    return pairs


def train_sentences(person):
    """예시 베낌 시험용: 목표 인물 코퍼스의 문장(검색 예시 역할)."""
    info = json.loads((ROOT / "data/eval/split_info.json").read_text(encoding="utf-8"))
    held = {u["user"]: set(u["heldout_docs"]) for u in info["users"]}[person]
    docs = [json.loads(l) for l in (ROOT / f"data/raw/corpus_{person}_raw.jsonl").open(encoding="utf-8") if l.strip()]
    return [d["text"] for d in docs if d["id"] not in held]


# ---------------------------------------------------------------- 오류 주입 (적용 못 하면 None)
def _sents(t):
    return kv._analyze(t)["sents"]


_NEG_SWAPS = [(r"(?<!고 )있다\.$", "없다."), (r"(?<!고 )있습니다\.$", "없습니다."), (r"없다\.$", "있다."),
              (r"없습니다\.$", "있습니다."), (r"한다\.$", "하지 않는다."), (r"된다\.$", "되지 않는다."),
              (r"했다\.$", "하지 않았다."), (r"되었다\.$", "되지 않았다."), (r"합니다\.$", "하지 않습니다."),
              (r"됩니다\.$", "되지 않습니다."), (r"했습니다\.$", "하지 않았습니다."), (r"이다\.$", "이 아니다."),
              (r"입니다\.$", "이 아닙니다."), (r"지 않는다\.$", "는다."), (r"지 않았다\.$", "었다.")]
_HEDGE_ADD = [(r"이다\.$", "일 수도 있다."), (r"한다\.$", "할 수도 있다."), (r"된다\.$", "될 수도 있다."),
              (r"었다\.$", "었던 것 같다."), (r"였다\.$", "였던 것 같다."), (r"있다\.$", "있을 수도 있다."),
              (r"없다\.$", "없을 수도 있다."), (r"합니다\.$", "할 수도 있습니다."), (r"됩니다\.$", "될 수도 있습니다."),
              (r"입니다\.$", "일 수도 있습니다."), (r"었습니다\.$", "었던 것 같습니다.")]
_HEDGE_DEL = [(r" 수 있다\.$", " 것이다."), (r" 수 있습니다\.$", " 것입니다."), (r" 수도 있다\.$", " 것이다."),
              (r" 것 같다\.$", " 것이다."), (r" 것 같습니다\.$", " 것입니다."), (r" 수 있으며", " 것이며"), (r" 수 있고", " 것이고")]


def _swap_sentence(t, rng, rules):
    ss = _sents(t)
    order = list(range(len(ss)))
    rng.shuffle(order)
    for i in order:
        for pat, rep in rules:
            if re.search(pat, ss[i]):
                return t.replace(ss[i], re.sub(pat, rep, ss[i], count=1), 1)
    return None


def inj_negation(t, rng, **_):
    return _swap_sentence(t, rng, _NEG_SWAPS)


def inj_number(t, rng, **_):
    ms = list(re.finditer(r"\d+", t))
    if not ms:
        return None
    m = rng.choice(ms)
    return t[:m.start()] + str(int(m.group()) + rng.choice([1, 2, 10])) + t[m.end():]


def inj_omission(t, rng, **_):
    ss = [s for s in _sents(t) if len(kv._lemmas(kv._analyze(s)["toks"])) >= 4]
    if len(_sents(t)) < 3 or not ss:
        return None
    s = rng.choice(ss)
    return re.sub(r"\s{2,}", " ", t.replace(s, "", 1)).strip()


def _insert_sentence(t, rng, sent):
    ss = _sents(t)
    anchor = rng.choice(ss)
    return t.replace(anchor, anchor + " " + sent, 1)


def inj_addition(t, rng, other_texts, **_):
    cands = [s for o in other_texts for s in _sents(o) if 25 <= len(s) <= 120 and s not in t]
    return _insert_sentence(t, rng, rng.choice(cands)) if cands else None


def inj_example_copy(t, rng, example_texts, **_):
    cands = [s for o in example_texts for s in _sents(o) if 25 <= len(s) <= 120 and s not in t]
    return _insert_sentence(t, rng, rng.choice(cands)) if cands else None


def inj_modality(t, rng, **_):
    """추측 → 단정(있으면 먼저), 없으면 단정 → 추측."""
    return _swap_sentence(t, rng, _HEDGE_DEL) or _swap_sentence(t, rng, _HEDGE_ADD)


def inj_scope(t, rng, **_):
    ss = _sents(t)
    rng.shuffle(ss)
    for s in ss:
        toks = kv._analyze(s)["toks"]
        for f, tag in toks:
            if tag == "NNG" and len(f) >= 2 and (f + "은" in s or f + "는" in s or f + "이" in s or f + "가" in s):
                return t.replace(s, s.replace(f, "모든 " + f, 1), 1)
    return None


def inj_entity(t, rng, **_):
    ents = [f for f, tag in kv._analyze(t)["toks"] if tag in ("SL", "NNP") and len(f) >= 2]
    if not ents:
        return None
    e = rng.choice(ents)
    repl = "Claude" if re.match(r"[A-Za-z]", e) else "아리스토텔레스"
    return t.replace(e, repl)


def inj_meta(t, rng, **_):
    return rng.choice(["다음은 요청하신 문체로 다시 쓴 글입니다.\n\n", "다시 쓴 글: "]) + t


INJECTIONS = {  # 이름: (함수, 잡혀야 하는 유형들)
    "부정 뒤집기": (inj_negation, {"NEGATION_FLIP"}),
    "숫자 바꾸기": (inj_number, {"NUMBER_MISMATCH"}),
    "문장 빼기": (inj_omission, {"OMISSION", "LOW_RECALL"}),
    "다른 글 문장 넣기": (inj_addition, {"ADDITION", "EXAMPLE_COPY"}),
    "예시 문장 베끼기": (inj_example_copy, {"EXAMPLE_COPY"}),
    "추측↔단정": (inj_modality, {"MODALITY_SHIFT"}),
    "범위어 '모든' 넣기": (inj_scope, {"SCOPE_CHANGE"}),
    "고유명사 바꾸기": (inj_entity, {"ENTITY_MISSING"}),
    "설명 머리말": (inj_meta, {"META_TEXT"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    pairs = legit_pairs()
    rng = random.Random(SEED)

    print("== ① 정상 쌍 오탐 (PASS가 아닌 비율; minor는 PASS) ==")
    base = {}
    by_set = defaultdict(Counter)
    for p in pairs:
        v = kv.verify(p["src"], p["out"])
        base[p["id"]] = v
        by_set[p["set"]][v["verdict"]] += 1
        if a.show and v["verdict"] != "PASS":
            print(f"  {p['id']}: " + " / ".join(f"{i['type']}({i['note'][:60]})" for i in v["issues"] if i["severity"] != "minor"))
    for name, c in by_set.items():
        n = sum(c.values())
        print(f"  {name:<10} n={n:>2}  PASS {c['PASS']}  REVIEW {c['REVIEW']}  FAIL {c['FAIL']}  → 오탐 {(n - c['PASS']) / n:.0%}")
    types = Counter(i["type"] for v in base.values() if v["verdict"] != "PASS" for i in v["issues"] if i["severity"] != "minor")
    print("  걸린 사유: " + ", ".join(f"{k} {c}" for k, c in types.most_common()))

    print("\n== ② 오류 주입 탐지율 (넣은 유형이 잡힘 / 적용 가능 건수) ==")
    texts_by_person = defaultdict(list)
    for p in pairs:
        texts_by_person[p["person"]].append(p["out"])
    ex_cache = {}
    sets = list(dict.fromkeys(p["set"] for p in pairs))
    print(f"  {'오류':<14}" + "".join(f"{s:>16}" for s in sets) + f"{'전체':>14}")
    for name, (fn, expect) in INJECTIONS.items():
        hit, n = Counter(), Counter()
        misses = []
        for p in pairs:
            others = [t for who, ts in texts_by_person.items() if who != p["person"] for t in ts]
            if p["person"] not in ex_cache:
                ex_cache[p["person"]] = train_sentences(p["person"])
            exs = rng.sample(ex_cache[p["person"]], 4)
            bad = fn(p["out"], rng, other_texts=others, example_texts=exs)
            if not bad or bad == p["out"]:
                continue
            v = kv.verify(p["src"], bad, examples=exs if name == "예시 문장 베끼기" else None)
            got = bool({i["type"] for i in v["issues"]} & expect)
            n[p["set"]] += 1
            hit[p["set"]] += got
            if not got:
                misses.append(p["id"])
        cells = [f"{hit[s]}/{n[s]} ({hit[s] / n[s]:.0%})" if n[s] else "-" for s in sets]
        tot = sum(n.values())
        print(f"  {name:<14}" + "".join(f"{c:>16}" for c in cells) + f"{sum(hit.values()) / tot:>13.0%}" if tot else "")
        if a.show and misses:
            print(f"      놓친 예: {misses[:3]}")
    print("\n읽는 법: ①은 낮을수록, ②는 높을수록 좋다. ②의 오류는 규칙으로 넣은 쉬운 오류라 실제 성능의 상한 쪽이다.")


if __name__ == "__main__":
    main()
