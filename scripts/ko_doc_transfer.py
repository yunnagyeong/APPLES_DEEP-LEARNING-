"""한→한 문체 변환, 글 한 편 단위 실험 (D-072). 입력은 외부 글(또는 AI 생성 글), 중립화 단계 없음.

입력 파일: data/ko_docs/<세트>/<이름>.txt  (세트 = select 선택용 / confirm 확인용). 해라체('-다') 2,000자 이상 권장.
  각 글을 목표 3명 × 조건 3(P·R·P+R)으로 변환한다. 변형(variant)마다 results/ko_doc/<변형>/ 에 따로 저장(이어서 실행 가능).

변환 강도 시험(D-072 ④, 선택용에서 한 번에 하나만 바꿔 변형 이름을 달리해 비교):
  기본        : python scripts/ko_doc_transfer.py gen --set select --variant base
  ① 모델      : LINGUA_GENERATION_MODEL=<flash 모델> python scripts/ko_doc_transfer.py gen --set select --variant flash
  ② 예시      : --k 8  /  --expand 2 (검색된 예시 뒤의 같은 문서 청크를 이어 붙임)   (변형 이름 예: k8, ex2)
  ③ 한 줄     : --boost  ("뜻과 형식은 유지하되 단어·어미·연결 표현은 이 필자처럼 적극적으로 바꿔도 된다")
  ④ 문단 나눔 : --unit para  (문단별 변환 후 다시 잇기, 예시는 글 전체 기준으로 같은 것)
  API 없이 프롬프트 확인: --dry

보고: python scripts/ko_doc_transfer.py report --set select --variant base flash ...
  지표(주): 같은 글의 세 출력에 세 사람을 하나씩 배정(점수 = 출력 점수 − 입력 점수 합 최대) → 맞게 배정된 비율, 우연 1/3.
            ※ 입력 점수 항은 세 사람 전체 합에서 상수라 배정 결과는 같음(식은 D-072 정의 그대로 둠).
  함께 보고: 단순 argmax 정답률, 입력 자체의 판정(바닥), 형식 검사 통과율, 기능 형태소 변화율.
  분류기: SVM(문단 학습, 글 전체 입력)이 주, Delta는 참고(D-072). 분류기로 예시를 고르지 않는다(D-068).
"""
import argparse, itertools, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("LINGUA_CORPUS_SPLIT", "train")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import ko_style
from src.ko_style import PEOPLE
import ko_style_transfer as kst  # read_jsonl, run_commit, model_info, is_429, heldout_docs 재사용

CONDS = ["P", "R", "P+R"]  # gen 기본값은 R만(문체 지시 없이 RAG만, D-096). P·P+R은 --conds로 비교할 때만
IN_DIR, OUT_DIR = ROOT / "data" / "ko_docs", ROOT / "results" / "ko_doc"
MIN_CHARS = 2000
SHORT = {"ganghee": "강희", "nakyung": "나경", "suyoung": "수영"}


def load_inputs(set_name):
    d = IN_DIR / set_name
    docs = {}
    for p in sorted(d.glob("*.txt")):
        t = p.read_text(encoding="utf-8").strip()
        if t: docs[p.stem] = t
    return docs


def paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def expand_example(ex, user, n):
    """검색된 예시 뒤에 같은 문서의 다음 청크 n개를 이어 붙인다(예시 길이 늘리기)."""
    if n <= 0: return ex
    from src.store import read_examples
    by_id = {e["id"]: e for e in read_examples(user)}
    doc, _, num = ex["id"].rpartition("_")
    texts, parts = [ex["text"]], [ex["id"]]
    for j in range(1, n + 1):
        nid = f"{doc}_{int(num) + j:0{len(num)}d}"
        if nid in by_id: texts.append(by_id[nid]["text"]); parts.append(nid)
    return {**ex, "id": "+".join(parts), "text": "\n".join(texts)}


def input_meta():
    """scripts/ko_make_inputs.py가 남긴 입력 글의 출처(주제를 가져온 코퍼스 문서). 없으면 빈 값."""
    p = IN_DIR / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def source_exclude(doc_name, user):
    """입력 주제의 출처 문서가 목표 사람의 글이면 검색에서 뺀다(같은 내용의 원문을 예시로 받아 베끼는 것 방지)."""
    m = input_meta().get(doc_name)
    return {m["doc"]} if m and m["person"] == user else set()


_PAIR_EMB = {}


def retrieve_pairs(text, user, k, exclude):
    """짝 예시(data/ko_pairs/pairs_<사람>.jsonl, 검증 통과분)에서 입력과 AI 쪽이 비슷한 짝 k개. 임베딩은 캐시."""
    import hashlib
    import numpy as np
    from src.embeddings import embed_text, embed_texts
    pairs = kst.read_jsonl(ROOT / "data" / "ko_pairs" / f"pairs_{user}.jsonl")
    pairs = [p for p in pairs if p["doc"] not in exclude]
    if not pairs: sys.exit(f"[중단] {user}의 짝 예시가 없습니다(scripts/ko_pairs.py check 먼저).")
    cache_p = ROOT / "data" / "ko_pairs" / "emb_cache.json"
    if not _PAIR_EMB and cache_p.exists(): _PAIR_EMB.update(json.loads(cache_p.read_text(encoding="utf-8")))
    h = lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]
    need = [p for p in pairs if h(p["ai"]) not in _PAIR_EMB]
    if need:
        for p, v in zip(need, embed_texts([p["ai"] for p in need])): _PAIR_EMB[h(p["ai"])] = v
        cache_p.write_text(json.dumps(_PAIR_EMB), encoding="utf-8")
    q = np.array(embed_text(text[:2000], task_type="RETRIEVAL_QUERY"))
    sc = [(float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v))), p) for p in pairs for v in [np.array(_PAIR_EMB[h(p["ai"])])]]
    sc.sort(key=lambda x: -x[0])
    return [{"id": p["pair_id"], "score": round(s, 4), "type": "짝", "text": p["original"], "ai": p["ai"]} for s, p in sc[:k]]


def retrieve(text, user, k, expand, exclude=frozenset(), pairs=False):
    if pairs:
        return retrieve_pairs(text, user, k, set(exclude) | kst.heldout_docs(user))
    from src.retrieve import retrieve_similar
    ex = retrieve_similar(text[:2000], user, k=k, exclude_doc_ids=kst.heldout_docs(user) | set(exclude))
    bad = [x["id"] for x in ex if x["id"].rsplit("_", 1)[0] in kst.heldout_docs(user)]
    if bad: sys.exit(f"[중단] 평가 문서가 RAG에 섞임: {bad}")
    ex = [{"id": x["id"], "score": round(float(x["score"]), 4), "type": x.get("type"), "text": x["text"]} for x in ex]
    return [expand_example(x, user, expand) for x in ex]


def gen(a):
    docs = load_inputs(a.set)
    if not docs: sys.exit(f"[중단] {IN_DIR / a.set}/*.txt 가 없습니다. 글을 넣고 다시 실행하세요.")
    short = [k for k, t in docs.items() if len(t) < MIN_CHARS]
    if short: print(f"[주의] {MIN_CHARS}자 미만 글: {', '.join(short)} (분류기 길이 곡선상 상한이 낮아짐)")
    out = OUT_DIR / a.variant; out.mkdir(parents=True, exist_ok=True)
    GEN, RET = out / "gen.jsonl", out / "retrieved.json"
    ret = json.loads(RET.read_text(encoding="utf-8")) if RET.exists() else {}
    done = {(r["set"], r["doc"], r["target"], r["cond"]) for r in kst.read_jsonl(GEN) if r.get("output") and not r.get("error")}
    todo = [(d, t, c) for d in docs for t in PEOPLE for c in a.conds if (a.set, d, t, c) not in done]
    calls = len(todo) * (len(paragraphs(next(iter(docs.values())))) if a.unit == "para" else 1)
    cfg = dict(unit=a.unit, k=a.k, expand=a.expand, boost=a.boost, pairs=a.pairs, version="new")
    commit, info = kst.run_commit(), (kst.model_info() if not a.dry else {})
    print(f"[변형 {a.variant}] {cfg} / 커밋 {commit} / 남은 (글,목표,조건) {len(todo)}개 (대략 호출 {calls}회 안팎)")
    if a.dry:
        d, t, c = (todo or [(next(iter(docs)), PEOPLE[0], "P+R")])[0]
        text = docs[d]; unit = paragraphs(text)[0] if a.unit == "para" else text
        lines, mode = ko_style.p_lines_for(t, unit)
        ex = [{"text": f"(예시 본문 — {t}의 학습용 글 {a.k}개{'(+이웃 청크 ' + str(a.expand) + ')' if a.expand else ''})"}] if "R" in c else None
        if ex and a.pairs: ex = [{"ai": "(AI가 다시 쓴 문단)", "text": f"({t}이 실제로 쓴 같은 문단)"}] * a.k
        print(f"=== {d} → {t} · {c} · 지시 {len(lines)}줄 (판단: {mode}) ===")
        print(ko_style.build_prompt(unit if len(unit) < 600 else unit[:600] + "…", lines if "P" in c else [], ex, "new", a.boost)); return
    consec = 0
    with GEN.open("a", encoding="utf-8") as f:
        for d, t, c in todo:
            key = f"{a.set}|{d}|{t}|{a.k}|{a.expand}" + ("|pairs" if a.pairs else "")
            if key not in ret:
                try: ret[key] = retrieve(docs[d], t, a.k, a.expand, source_exclude(d, t), a.pairs)
                except Exception as e: sys.exit(f"[중단] 검색 실패({key}): {type(e).__name__}: {str(e)[:200]}")
                RET.write_text(json.dumps(ret, ensure_ascii=False, indent=1), encoding="utf-8")
            rec = {"variant": a.variant, "set": a.set, "doc": d, "target": t, "cond": c, "chars": len(docs[d]), "commit": commit, **cfg, **info}
            try:
                units = paragraphs(docs[d]) if a.unit == "para" else [docs[d]]
                rs = [ko_style.convert_style(u, t, c, examples=ret[key], version="new", boost=a.boost) for u in units]
                rec.update(output="\n\n".join(r["converted"] for r in rs), lines=[r["lines"] for r in rs], gate_mode=[r["gate_mode"] for r in rs],
                           example_ids=[x["id"] for x in rs[0]["examples"]], prompts=[r["prompt"] for r in rs], error=""); consec = 0
            except Exception as e:
                rec.update(output="", error=f"{type(e).__name__}: {e}"[:300])
                consec = consec + 1 if kst.is_429(rec["error"]) else 0
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{a.variant}] {d} → {t} {c} " + ("ok" if rec["output"] else "ERR " + rec["error"][:80]))
            if consec >= 3: sys.exit("[중단] 429 연속 3회: 쿼터 확인 후 다시 실행하면 이어서 합니다.")


# ───────── 측정 ─────────
def doc_format(src, out, ko_format, ratio_max=0.2):
    """글 단위 형식 검사(규칙). 문단 수가 같으면 문단별로 검사해 표시 문단 비율이 ratio_max 초과일 때 표시, 다르면 바로 표시. 임계값은 초안."""
    sp, op = paragraphs(src), paragraphs(out)
    if len(sp) != len(op): return False, [f"문단 수 {len(sp)}→{len(op)}"]
    flagged = [ko_format.check(s, o) for s, o in zip(sp, op)]
    bad = [c for c in flagged if not c["pass"]]
    if len(bad) / max(1, len(sp)) > ratio_max:
        why = Counter(r.split(" ")[0] for c in bad for r in c["reasons"])
        return False, [f"표시 문단 {len(bad)}/{len(sp)} (" + ", ".join(f"{k} {v}" for k, v in why.most_common()) + ")"]
    return True, ([f"표시 문단 {len(bad)}/{len(sp)}(허용)"] if bad else [])


def func_change(src, out):
    """기능 형태소 변화율 = 1 - Σmin/Σmax (다중집합). 0 = 같음, 1 = 완전히 다름."""
    a, b = Counter(ko_style.func_tokens(src, True)), Counter(ko_style.func_tokens(out, True))
    keys = set(a) | set(b)
    mx = sum(max(a[k], b[k]) for k in keys)
    return 1 - sum(min(a[k], b[k]) for k in keys) / max(1, mx)


def best_assignment(S):
    """S[t][p] = 목표 t 출력의 사람 p 점수 변화. 세 출력에 세 사람을 하나씩 배정해 합이 최대인 순열."""
    best = max(itertools.permutations(PEOPLE), key=lambda perm: sum(S[t][p] for t, p in zip(PEOPLE, perm)))
    return dict(zip(PEOPLE, best))


def report(a):
    from src import ko_classify as kc, ko_format, ko_verify
    if not ko_style._kiwi(): sys.exit("[중단] kiwipiepy가 필요합니다(Mac의 .venv에서 실행).")
    docs = load_inputs(a.set)
    variants = a.variant or sorted(p.name for p in OUT_DIR.iterdir() if (p / "gen.jsonl").exists())
    clf = kc.Classifiers()
    print("== 글 단위 평가 (D-072) ==\n" + clf.sizes())
    in_sc = clf.scores(list(docs.values())); in_idx = {d: i for i, d in enumerate(docs)}
    in_pred = clf.predict(list(docs.values()))
    print(f"\n입력 {len(docs)}편(세트 {a.set}) 평균 {sum(map(len, docs.values())) // max(1, len(docs))}자 | 입력 판정(바닥)  SVM {dict(Counter(in_pred['SVM']))}  Delta {dict(Counter(in_pred['Delta']))}")
    rows = {}
    for v in variants:
        recs = {(r["doc"], r["target"], r["cond"]): r for r in kst.read_jsonl(OUT_DIR / v / "gen.jsonl")
                if r.get("set") == a.set and r.get("output") and not r.get("error") and r["doc"] in docs}
        conds = [c for c in CONDS if any(k[2] == c for k in recs)]  # 이 변형에서 실제로 돌린 조건만
        ds = [d for d in docs if all((d, t, c) in recs for t in PEOPLE for c in conds)]
        if not ds: print(f"\n[변형 {v}] 완성된 글 없음"); continue
        cfg = next(iter(recs.values()))
        print(f"\n=== 변형 {v}: 글 {len(ds)}/{len(docs)}편 완성 | unit={cfg.get('unit')} k={cfg.get('k')} expand={cfg.get('expand')} boost={cfg.get('boost')} 모델={cfg.get('model')} 커밋={cfg.get('commit')} ===")
        keys = [(d, t, c) for d in ds for t in PEOPLE for c in conds]
        out_sc = clf.scores([recs[k]["output"] for k in keys]); out_pred = clf.predict([recs[k]["output"] for k in keys])
        pos = {k: i for i, k in enumerate(keys)}
        fmt = {k: doc_format(docs[k[0]], recs[k]["output"], ko_format) for k in keys}
        print(f"{'조건':<5}{'배정SVM':<14}{'배정Delta':<14}{'argmaxSVM':<14}{'argmaxDelta':<14}{'형식통과':<10}{'형식통과 세트만 배정SVM':<24}{'기능형태소 변화율'}")
        for c in conds:
            res = {}
            for m in ("SVM", "Delta"):
                ok = tot = 0; ok_p = tot_p = 0
                for d in ds:
                    S = {t: {p: out_sc[m][pos[(d, t, c)]][p] - in_sc[m][in_idx[d]][p] for p in PEOPLE} for t in PEOPLE}
                    perm = best_assignment(S); hit = sum(perm[t] == t for t in PEOPLE)
                    ok += hit; tot += 3
                    if all(fmt[(d, t, c)][0] for t in PEOPLE): ok_p += hit; tot_p += 3
                res[m] = (ok, tot, ok_p, tot_p)
            am = {m: sum(out_pred[m][pos[(d, t, c)]] == t for d in ds for t in PEOPLE) for m in ("SVM", "Delta")}
            fp = sum(fmt[(d, t, c)][0] for d in ds for t in PEOPLE)
            fc = sum(func_change(docs[d], recs[(d, t, c)]["output"]) for d in ds for t in PEOPLE) / (3 * len(ds))
            n = 3 * len(ds)
            rows[(v, c)] = (res["SVM"][0] / n, res["Delta"][0] / n, am["SVM"] / n, fp / n, fc)
            sv, de = res["SVM"], res["Delta"]
            def cell(x, y): return f"{x}/{y} ({x / y:.2f})" if y else "-"
            fp_s = f"{fp}/{n}"
            print(f"{c:<5}{cell(sv[0], sv[1]):<14}{cell(de[0], de[1]):<14}{cell(am['SVM'], n):<14}{cell(am['Delta'], n):<14}{fp_s:<10}{cell(sv[2], sv[3]):<24}{fc:.3f}")
        # 의미 검증(강희 ko_verify, 규칙 기반·API 없음): 예시 본문은 retrieved.json에서 찾는다
        ret = json.loads((OUT_DIR / v / "retrieved.json").read_text(encoding="utf-8")) if (OUT_DIR / v / "retrieved.json").exists() else {}
        ver = {}
        for k in keys:
            r_ = recs[k]; rk = f"{a.set}|{k[0]}|{k[1]}|{r_.get('k')}|{r_.get('expand')}" + ("|pairs" if r_.get("pairs") else "")
            ver[k] = ko_verify.verify(docs[k[0]], r_["output"], ret.get(rk))
        print("\n  의미 검증 (규칙 기반, 판정은 보조 — 사람이 확인): PASS/REVIEW/FAIL · 입력 그대로(unchanged) · 엄격 배정(세 출력 모두 PASS인 글만, SVM)")
        for c in conds:
            vs = Counter(ver[(d, t, c)]["verdict"] for d in ds for t in PEOPLE)
            un = sum(bool(ver[(d, t, c)]["stats"].get("unchanged")) for d in ds for t in PEOPLE)
            okd = [d for d in ds if all(ver[(d, t, c)]["verdict"] == "PASS" for t in PEOPLE)]
            hit = 0
            for d in okd:
                S = {t: {p: out_sc["SVM"][pos[(d, t, c)]][p] - in_sc["SVM"][in_idx[d]][p] for p in PEOPLE} for t in PEOPLE}
                perm = best_assignment(S); hit += sum(perm[t] == t for t in PEOPLE)
            n = 3 * len(ds)
            print(f"  {c:<5}PASS {vs['PASS']}/{n} REVIEW {vs['REVIEW']}/{n} FAIL {vs['FAIL']}/{n} | unchanged {un}/{n} | 엄격 배정 {hit}/{3 * len(okd)}" + (f" ({hit / (3 * len(okd)):.2f}, 글 {len(okd)}편)" if okd else " (모두 PASS인 글 없음)"))
        for k, x in ver.items():
            if x["verdict"] != "PASS": print(f"    {x['verdict']} {k[0]}→{SHORT[k[1]]} {k[2]}: {ko_verify.summarize(x)}"[:220])
        flagged = [(k, w) for k, (ok, w) in fmt.items() if not ok]
        print(f"  형식 표시 {len(flagged)}/{len(keys)}건" + (": " + "; ".join(f"{k[0]}→{SHORT[k[1]]} {k[2]} {'/'.join(w)}" for k, w in flagged[:8]) + (" …" if len(flagged) > 8 else "") if flagged else ""))
    print("\n읽는 법: 우연 = 1/3. 배정 지표는 같은 글의 세 출력을 서로 다른 사람에게 배정하는 방식이라 분류기가 한 사람으로 쏠려도 영향을 덜 받음(D-072). "
          "출력을 본 뒤 고른 지표이므로 선택용에서는 가설, 확인용 측정 전에 주 지표로 고정하고 argmax 정답률·입력 판정(바닥)을 같이 보고. 글 하나가 3개 출력이라 서로 독립이 아님. "
          "형식 임계값은 초안. 강도를 올릴수록 뜻이 바뀔 위험이 있어 사람이 의미 유지를 확인해야 함. 분류기로 예시를 고르지 않음.")
    if len(variants) > 1:
        print("\n변형 비교 (배정SVM / argmaxSVM / 형식통과율 / 기능형태소 변화율)")
        for (v, c), (asg, _, am, fp, fc) in rows.items(): print(f"  {v:<10}{c:<5} {asg:.2f} / {am:.2f} / {fp:.2f} / {fc:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("--set", default="select", choices=["select", "confirm"]); g.add_argument("--variant", required=True)
    g.add_argument("--unit", choices=["doc", "para"], default="doc"); g.add_argument("--k", type=int, default=4); g.add_argument("--expand", type=int, default=0)
    g.add_argument("--boost", action="store_true"); g.add_argument("--dry", action="store_true")
    g.add_argument("--pairs", action="store_true", help="R 예시를 짝 예시(AI 글 ↔ 이 필자의 글)로 (D-078)")
    g.add_argument("--conds", nargs="+", choices=CONDS, default=["R"], help="돌릴 조건(기본 R만, D-096)")
    r = sp.add_parser("report"); r.add_argument("--set", default="select", choices=["select", "confirm"]); r.add_argument("--variant", nargs="*")
    a = ap.parse_args(); {"gen": gen, "report": report}[a.cmd](a)
