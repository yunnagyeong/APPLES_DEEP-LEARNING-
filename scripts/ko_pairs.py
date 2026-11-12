"""짝 데이터 (AI가 다시 쓴 문단 → 그 사람의 실제 원문) 만들기·거르기·분류기 상한 확인 (D-078·D-080~D-082).

  1) select   : 사람마다 학습 문서(평가 문서 제외, essay·report)를 시드로 섞어 앞에서부터 고르고, 그 문서의
                150~450자 문단을 사람당 15개 고른다. 짝에 쓴 문서는 분류기 학습에서 뺀다(D-081 문제 1).
                → data/ko_pairs/selected.json  (API 없음)
  2) gen      : 고른 문단을 AI 말투로 다시 쓴다(해라체·문장 수·형식·내용 유지). → data/ko_pairs/raw.jsonl
  3) check    : 거르기(D-082). 판정 모델이 원문 claim마다 다시 쓴 글의 근거 구간·오류 유형을 고르고(한→한 사본 프롬프트,
                verify.py 원본은 그대로), 규칙(숫자, 글자 수 비 0.8~1.25, 임베딩 유사도 하한)을 더해 문제 0개인 짝만 채택.
                재번역(최소 수정) 없음. --control: 알려진 덧붙임 사례(neutral.jsonl의 suyoung_doc_024_02)가 걸리는지 먼저 확인.
                → data/ko_pairs/pairs_<사람>.jsonl(채택분), data/ko_pairs/check.csv(전체 + 사람 O/X 칸, 표본 표시)
  4) ceiling  : 짝 문서를 뺀 분류기의 원문 상한(평가 문서 문단·글, 글 배정 양성 대조). API 없음.

판정 모델은 분류만 하고, 채택 여부는 아래 고정 규칙이 정한다(원칙 1). 사람 확인 표본은 사람당 4개.
"""
import argparse, csv, itertools, json, random, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "ko_pairs"
SEL, RAW = OUT / "selected.json", OUT / "raw.jsonl"
from src.ko_style import PEOPLE  # LINGUA_PEOPLE로 바꿀 수 있음
PER_PERSON, MIN_C, MAX_C, SEED, HUMAN_SAMPLE = 15, 150, 450, "pairs", 4
RATIO_LO, RATIO_HI = 0.8, 1.25
MIN_CHANGE = 0.15  # 기능 형태소 변화율 하한(초안): 거의 복사한 짝은 변환 방식을 보여 주지 못함


def read_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


MAX_ATTEMPTS = 3


def latest(rows):
    """pair_id마다 가장 마지막 시도(attempt)만."""
    out = {}
    for r in rows:
        if r.get("ai") and r.get("attempt", 1) >= out.get(r["pair_id"], {}).get("attempt", 0):
            out[r["pair_id"]] = r
    return out


def ckey(r):
    return f"{r['pair_id']}#{r.get('attempt', 1)}"


def hapsyo_rate(text):
    """종결어미 중 '-습니다/-ㅂ니다' 비율(Kiwi)."""
    from src.ko_style import _kiwi
    ef = [t.form for t in _kiwi().tokenize(text) if t.tag == "EF"]
    return sum("니다" in f for f in ef) / max(1, len(ef))


def heldout():
    info = json.loads(__import__("src.config", fromlist=["x"]).SPLIT_INFO_PATH.read_text(encoding="utf-8"))
    return {u["user"]: set(u["heldout_docs"]) for u in info["users"]}


def paras(text):
    return [p.strip() for p in text.split("\n\n") if MIN_C <= len(p.strip()) <= MAX_C]


def commit():
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    d = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts/ko_pairs.py"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return h + (" (+미커밋 변경 있음)" if d else "")


# ───────── 1) select ─────────
def select(_):
    from src import config
    ho, out = heldout(), {"pair_docs": {}, "paragraphs": []}
    for u in PEOPLE:
        docs = []
        for l in config.RAW_DIR.joinpath(f"corpus_{u}_raw.jsonl").open(encoding="utf-8"):
            if not l.strip(): continue
            o = json.loads(l)
            if o["id"] in ho[u] or o.get("type") not in ("essay", "report"): continue
            ps = paras(o["text"])
            if ps: docs.append((o["id"], o["type"], ps))
        docs.sort(key=lambda d: d[0]); random.Random(f"{SEED}-{u}").shuffle(docs)
        chosen, pool = [], []
        for d in docs:
            if len(pool) >= PER_PERSON: break
            chosen.append(d[0]); pool += [(d[0], d[1], i, p) for i, p in enumerate(d[2])]
        random.Random(f"{SEED}-{u}-p").shuffle(pool)
        out["pair_docs"][u] = sorted(chosen)
        out["paragraphs"] += [{"pair_id": f"{u}_{doc}_p{i:02d}", "person": u, "doc": doc, "type": t, "original": p}
                              for doc, t, i, p in sorted(pool[:PER_PERSON], key=lambda x: (x[0], x[2]))]
        print(f"{u}: 학습 문서 {len(docs)}편 중 짝용 {len(chosen)}편 {chosen} → 문단 {min(PER_PERSON, len(pool))}개")
    OUT.mkdir(parents=True, exist_ok=True)
    SEL.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {SEL.relative_to(ROOT)} (짝 문서는 분류기 학습에서 뺄 것: ceiling 참고)")


# ───────── 2) gen ─────────
def rewrite_prompt(text):
    # v2(D-087): v1은 "그대로 둘 것"이 많아 거의 복사(기능 형태소 변화 0.02~0.04)가 됨 → 표현은 크게 바꾸라고 명시
    return ("다음 글을 평범한 AI가 쓴 것 같은 한국어로 다시 써 주세요. 뜻은 그대로 두고 표현은 크게 바꾸세요.\n\n"
            "반드시 바꿀 것: 문장마다 단어, 어미, 조사, 연결 표현을 원문과 다르게 고쳐 쓰세요. 원문의 구절을 그대로 옮기지 말고, "
            "AI가 흔히 쓰는 무난하고 일반적인 표현을 쓰세요.\n"
            "그대로 둘 것: 뜻과 내용(새로운 사실·이유·예시·꾸밈말을 더하거나 빼지 않음), 확신 정도, 숫자와 고유명사, "
            "문장 순서와 문장 수, 형식(줄글·목록), '-다'로 끝나는 말투.\n"
            "다시 쓴 글만 출력하세요.\n\n원문:\n" + text)


def gen(a):
    from src import config, ko_style
    if not SEL.exists(): sys.exit("[중단] 먼저 select를 실행하세요.")
    sel = json.loads(SEL.read_text(encoding="utf-8"))["paragraphs"]
    last = latest(read_jsonl(RAW))
    if getattr(a, "retry", False):  # 탈락한 짝만 새로 다시 생성(시도 번호 +1, 최대 MAX_ATTEMPTS)
        chk = {r["key"]: r for r in read_jsonl(OUT / "check.jsonl") if "key" in r}
        todo = []
        for p in sel:
            r = last.get(p["pair_id"])
            if r and not chk.get(ckey(r), {}).get("accept", True) and r.get("attempt", 1) < MAX_ATTEMPTS:
                todo.append({**p, "attempt": r.get("attempt", 1) + 1})
    else:
        todo = [{**p, "attempt": 1} for p in sel if p["pair_id"] not in last]
    print(f"[커밋] {commit()} / 모델 {config.GENERATION_MODEL} / 남은 {len(todo)}개")
    if a.dry:
        print(rewrite_prompt(todo[0]["original"] if todo else sel[0]["original"])); return
    consec = 0
    with RAW.open("a", encoding="utf-8") as f:
        for p in todo:
            rec = {**p, "model": config.GENERATION_MODEL, "commit": commit()}
            try:
                rec["ai"] = ko_style.generate(rewrite_prompt(p["original"]), "짝 재작성"); rec["error"] = ""; consec = 0
            except Exception as e:
                rec.update(ai="", error=f"{type(e).__name__}: {e}"[:300])
                consec = consec + 1 if ("429" in rec["error"] or "RESOURCE_EXHAUSTED" in rec["error"]) else 0
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"{p['pair_id']} " + (f"ok (길이 비 {len(rec['ai']) / len(p['original']):.2f})" if rec["ai"] else "ERR " + rec["error"][:80]))
            if consec >= 3: sys.exit("[중단] 429 연속 3회. 다시 실행하면 이어서 합니다.")


# ───────── 3) check ─────────
KO_JUDGE_PROMPT = """You are a meticulous Korean meaning-fidelity verifier.
Your ONLY job is meaning fidelity between a Korean ORIGINAL and a Korean REWRITE. The rewrite deliberately changed
the writing style, so differences in wording, endings, connectives, register and rhythm are EXPECTED and are NOT errors.

STEP 1 - Original claims.
{claim_step}

STEP 2 - Align each original claim to the rewrite.
Quote the exact span of the rewrite that expresses it as "evidence" ("" if absent), then choose ONE label:
- OK: same meaning (paraphrase, synonym, register or style change is fine)
- TENSE_SHIFT: only the time reference changed
- INTENSITY_SHIFT: degree slightly stronger/weaker, claim otherwise the same
- MODALITY_SHIFT: certainty or possibility changed (수 있다/것 같다/보인다 <-> 단정), including a hedge removed or added
- ATTRIBUTION_LOST: subjective framing (생각한다, 내 생각엔, 개인적으로) removed, turned into fact, or added
- SCOPE_CHANGE: quantity or scope changed (some -> all, often -> always, adding/removing "only")
- CONDITION_DROPPED: a condition or restriction was removed or changed
- OMISSION: the claim is missing from the rewrite
- NEGATION_FLIP: negation reversed
- NUMBER_CHANGE: a number, percentage, date or quantity changed
- ENTITY_CHANGE: a name, place, organization or technical term changed into a different one
- COMPARISON_REVERSED: comparison direction changed
- CAUSAL_CHANGE: cause and effect reversed, correlation turned into causation, or a causal link invented
- CONTRADICTION: any other reversal of meaning

STEP 3 - Additions.
List every span of the rewrite that adds content not present in the original:
- MINOR_ADDITION: a word that adds no new proposition (intensifier, evaluative modifier, connective), e.g. "대표적인", "적극적인"
- UNSUPPORTED_ADDITION: a new proposition, reason, explanation, justification, example or fact
Do not list pure style changes. If nothing was added, return an empty list.

Write every "note" as one short sentence in Korean ("" for OK).
Return ONLY JSON:
{{"source_claims": [{{"claim": "...", "evidence": "...", "label": "OK", "note": ""}}],
  "additions": [{{"text": "...", "label": "MINOR_ADDITION", "note": "..."}}]}}

Korean original:
{source}

Korean rewrite:
{translation}
"""


def ko_number_issues(src, out):
    from src import verify
    a, b = verify._ko_numbers(src), verify._ko_numbers(out)
    iss = []
    if a - b: iss.append({"type": "NUMBER_CHECK", "note": f"원문 숫자 {', '.join(sorted(a - b))}이(가) 다시 쓴 글에 없음"})
    if b - a: iss.append({"type": "NUMBER_CHECK", "note": f"다시 쓴 글에 원문에 없는 숫자 {', '.join(sorted(b - a))}"})
    return iss


def func_change(a, b):
    """두 글의 기능 형태소 분포 차이(0=같음, 1=전부 다름). 지시 표현 제외 버전."""
    from collections import Counter
    from src.ko_style import func_tokens
    A, B = Counter(func_tokens(a, True)), Counter(func_tokens(b, True))
    return sum(((A - B) + (B - A)).values()) / max(1, sum(A.values()) + sum(B.values()))


def judge_pair(original, ai):
    from src import config, llm, verify
    from src.embeddings import embed_texts
    prompt = KO_JUDGE_PROMPT.format(claim_step=verify._SPLIT_STEP.replace("Korean source", "Korean original"),
                                    source=original.strip(), translation=ai.strip())
    judgement, model = llm.generate(prompt, model=config.VERIFY_MODEL, fallback_model=config.VERIFY_FALLBACK_MODEL,
                                    thinking_level=config.VERIFY_THINKING_LEVEL, json_mode=True,
                                    label="짝 판정", validate=verify._validate_judgement)
    try:
        e1, e2 = embed_texts([original, ai]); sim = verify.cosine_similarity(e1, e2)
    except Exception as e:
        print(f"  [유사도 실패, 건너뜀] {e}"); sim = None
    claims, adds, issues = verify.build_issues(judgement, ko_number_issues(original, ai), sim)
    ratio = len(ai) / max(1, len(original))
    change = func_change(original, ai)
    if hapsyo_rate(original) == 0 and hapsyo_rate(ai) > 0:
        issues.append(verify._issue("REGISTER_CHANGE", "rule", note="원문은 '-다'인데 다시 쓴 글에 '-습니다'가 섞임(말투 변화는 개인 문체가 아님)"))
    if change < MIN_CHANGE:
        issues.append(verify._issue("TOO_SIMILAR", "rule", note=f"기능 형태소 변화율 {change:.2f} < {MIN_CHANGE} (거의 복사)"))
    if not RATIO_LO <= ratio <= RATIO_HI:
        issues.append(verify._issue("LENGTH_RATIO", "rule", note=f"글자 수 비 {ratio:.2f} (허용 {RATIO_LO}~{RATIO_HI})"))
    return {"verdict": verify.decide_verdict(issues), "accept": not issues, "issues": issues, "similarity": sim,
            "ratio": round(ratio, 3), "change": round(change, 3), "n_claims": len(claims), "judge_model": model}


def check(a):
    if a.control:
        ctl = [r for r in read_jsonl(ROOT / "results/ko_style/neutral.jsonl") if r["nid"] == "suyoung_doc_024_02"]
        if not ctl: sys.exit("[중단] 양성 대조 문단이 없습니다(results/ko_style/neutral.jsonl).")
        r = judge_pair(ctl[0]["source"], ctl[0]["neutral"])
        print(f"[양성 대조] suyoung_doc_024_02 → 채택 {r['accept']} / 판정 {r['verdict']}")
        for i in r["issues"]: print(f"  - {i['type']} ({i['severity']}): {i.get('evidence') or ''} {i.get('note') or ''}")
        print("  기대: 덧붙임('대표적인'·'적극적인'·'부활에 성공하여' 등)이 걸려 채택 False" + ("  ✔" if not r["accept"] else "  ✘ 걸리지 않음 → 거르기 기준 재검토"))
        return
    rows = list(latest(read_jsonl(RAW)).values())
    if not rows: sys.exit("[중단] 먼저 gen을 실행하세요.")
    res_p = OUT / "check.jsonl"
    done = {r["key"]: r for r in read_jsonl(res_p) if "key" in r}
    with res_p.open("a", encoding="utf-8") as f:
        for r in rows:
            if ckey(r) in done: continue
            try:
                j = judge_pair(r["original"], r["ai"]); j["error"] = ""
            except Exception as e:
                j = {"accept": False, "verdict": "ERROR", "issues": [], "error": f"{type(e).__name__}: {e}"[:300]}
            rec = {"key": ckey(r), "pair_id": r["pair_id"], "attempt": r.get("attempt", 1), **j}; done[ckey(r)] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"{ckey(r)} 채택 {j['accept']} {j['verdict']} " + "; ".join(i["type"] for i in j["issues"])[:120])
    by = {r["pair_id"]: r for r in rows}
    acc = {u: [] for u in PEOPLE}
    for pid, r in by.items():
        if done.get(ckey(r), {}).get("accept"): acc[r["person"]].append(r)
    for u in PEOPLE:
        (OUT / f"pairs_{u}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in acc[u]), encoding="utf-8")
    sample = {x["pair_id"] for u in PEOPLE for x in random.Random(f"{SEED}-{u}-h").sample(acc[u], min(HUMAN_SAMPLE, len(acc[u])))}
    with (OUT / "check.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "사람", "채택", "판정", "문제", "글자 수 비", "기능 형태소 변화율", "유사도", "원문", "AI가 다시 쓴 글", "사람 확인 표본", "사람: 뜻 유지(O/X)", "메모"])
        for pid, r in by.items():
            j = done.get(ckey(r), {})
            w.writerow([pid, r["person"], j.get("accept"), j.get("verdict"),
                        "; ".join(f"{i['type']}: {i.get('evidence') or i.get('note') or ''}" for i in j.get("issues", [])),
                        j.get("ratio"), j.get("change"), None if j.get("similarity") is None else round(j["similarity"], 3),
                        r["original"], r["ai"], "표본" if pid in sample else "", "", ""])
    tot = {u: sum(1 for x in rows if x["person"] == u) for u in PEOPLE}
    print("\n채택: " + ", ".join(f"{u} {len(acc[u])}/{tot[u]}" for u in PEOPLE))
    print(f"→ data/ko_pairs/pairs_<사람>.jsonl, data/ko_pairs/check.csv ('표본' 행을 사람이 O/X: 검증기와 사람 일치율용)")


# ───────── 4) ceiling ─────────
def ceiling(_):
    import numpy as np
    from src import ko_classify as kc
    from src.ko_style import func_tokens
    from src import config
    if not SEL.exists(): sys.exit("[중단] 먼저 select를 실행하세요.")
    pair_docs = {u: set(v) for u, v in json.loads(SEL.read_text(encoding="utf-8"))["pair_docs"].items()}
    ho = heldout()
    evd = []
    for u in PEOPLE:
        for l in config.RAW_DIR.joinpath(f"corpus_{u}_raw.jsonl").open(encoding="utf-8"):
            if l.strip() and json.loads(l)["id"] in ho[u]: evd.append((u, json.loads(l)["text"]))
    for name, drop in (("현재", {u: set() for u in PEOPLE}), ("짝 문서 뺌", pair_docs)):
        base = kc.load_corpus()
        tp, td = [], []
        for u in PEOPLE:
            for l in config.RAW_DIR.joinpath(f"corpus_{u}_raw.jsonl").open(encoding="utf-8"):
                if not l.strip(): continue
                o = json.loads(l)
                if o["id"] in ho[u] or o["id"] in drop[u] or o.get("type") not in ("essay", "report"): continue
                tp += [(u, p) for p in kc._paragraphs(o["text"])]; td.append({"author": u, "toks": func_tokens(o["text"], False)})
        C = kc.Classifiers({"train_docs": td, "train_paras": tp, "eval_paras": base["eval_paras"]})
        te = base["eval_paras"]; p = C.predict([t for _, t in te]); pd_ = C.predict([t for _, t in evd])
        cls = list(C.svm.classes_); ix = [cls.index(q) for q in PEOPLE]
        by = {u: [t for a, t in evd if a == u] for u in PEOPLE}; ok = []
        for trip in itertools.product(*[by[u] for u in PEOPLE]):
            S = C.svm.decision_function([C._feat(t) for t in trip])[:, ix]
            best = max(itertools.permutations(range(3)), key=lambda pm: sum(S[r, pm[r]] for r in range(3)))
            ok += [best[r] == r for r in range(3)]
        print(f"[{name}] 학습 문단 {len(tp)}개 | 평가 문단 SVM {kc.accuracy([a for a, _ in te], p['SVM']):.2f} | "
              f"평가 글 SVM {kc.accuracy([a for a, _ in evd], pd_['SVM']):.2f} Delta {kc.accuracy([a for a, _ in evd], pd_['Delta']):.2f} | 글 배정(양성 대조) {np.mean(ok):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("select").set_defaults(fn=select)
    g = sp.add_parser("gen"); g.add_argument("--dry", action="store_true")
    g.add_argument("--retry", action="store_true", help="탈락한 짝만 다시 생성(최대 3회 시도)"); g.set_defaults(fn=gen)
    c = sp.add_parser("check"); c.add_argument("--control", action="store_true"); c.set_defaults(fn=check)
    sp.add_parser("ceiling").set_defaults(fn=ceiling)
    a = ap.parse_args(); a.fn(a)
