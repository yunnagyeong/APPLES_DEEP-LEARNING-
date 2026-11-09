"""한국어→한국어 문체 변환 실험 (D-043·D-044). 생성만 호출하고 검증은 하지 않는다.

질문: 같은 중립문을 세 사람 각각의 문체로 바꾸게 했을 때, 문체 지시(P)·본인 글 예시(R)·둘 다(P+R)가
      그 사람 문체를 옮기는가? 한-영 번역에서 섞여 있던 "지시·예시가 문체를 옮길 수 있나"만 따로 본다.

흐름 (모두 results/ko_style/ 아래, 이어서 실행 가능):
  1) neutralize : 나경 split(seed 42) 평가 문서 청크(data/eval/eval_sources.jsonl)에서 사람당 4개(150~450자, 시드 고정)를
                  고르고, 생성 모델에게 "평범한 AI 말투로, 뜻은 그대로" 다시 쓰게 해 중립문 12개를 만든다(원문은 정답으로 보관).
  2) gen        : 중립문 12 × 목표 3명 × 조건 3(P / R / P+R) = 108회. N(중립 그대로)은 생성 없음.
                  공통 규칙: 내용 추가·삭제 금지, 확신 정도 유지, 문장 수 자유.
                  P  = 한국어 문체 지시만(중립문에 해당 기능이 있을 때만 넣음, 수영 문장 길이 줄은 항상)
                  R  = 목표 본인 글 예시 4개만(학습용 색인, 평가 문서는 제외)
                  P+R= 둘 다.   --dry 로 API 없이 프롬프트만 확인.
  3) report     : API 없음(kiwipiepy 필요). ① 3×3 작성자 귀속(기능 형태소 top100 Burrows' Delta, 학습용 문서 기준,
                  지시 그룹 표현 뺀 버전·넣은 버전) ② 본인 문단의 원문 거리(중립문 대비) ③ 사람이 볼 CSV 2개 ④ 분류기 두 개(선형 SVM·Delta, 문단/묶음 단위별 상한·N·P·R·P+R, scikit-learn 필요).

사용법 (저장소 루트, 실제 호출은 API 키가 있는 Mac에서):
  .venv/bin/python scripts/ko_style_transfer.py neutralize
  .venv/bin/python scripts/ko_style_transfer.py gen --dry            # 프롬프트만 확인
  .venv/bin/python scripts/ko_style_transfer.py gen
  .venv/bin/python scripts/ko_style_transfer.py report
다른 생성 모델로 돌리려면 LINGUA_GENERATION_MODEL 등 환경변수와 --tag <이름> (중립문은 기본 폴더 것을 복사해 씀).

주의: 지시에 쓴 기능어(그러나·따라서·므로 등)로 귀속을 재면 순환이라 report는 그 표현을 뺀 버전을 함께 낸다.
      중립화도 LLM이 하지만 판정이 아니라 입력 생성이다. 번역·변환 결과의 의미 확인과 블라인드 판단은 사람이 한다.
"""
import argparse, csv, json, os, random, re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("LINGUA_CORPUS_SPLIT", "train")  # config 가져오기 전에: RAG는 학습용 코퍼스만
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "ko_style"
NEUTRAL, GEN, RET = OUT / "neutral.jsonl", OUT / "gen.jsonl", OUT / "retrieved.json"
BASE_NEUTRAL = NEUTRAL

SEED, PER_PERSON, MIN_C, MAX_C = 42, 4, 150, 450
from src import ko_style
from src.ko_style import (PEOPLE, PROMPT_VERSIONS, build_prompt, build_ref, detect_functions, func_tokens,  # noqa: F401
                          load_reference_docs, p_lines_for, zvec)
CONDS = ["P", "R", "P+R"]

def versions_of(a):
    v = getattr(a, "version", "both")
    return list(PROMPT_VERSIONS) if v == "both" else [v]

def kiwi_ok():
    return ko_style._kiwi() is not None

def build_neutral_prompt(text):
    return ("다음 글을 평범한 AI 말투의 한국어로, 뜻과 내용은 그대로 두고 다시 써 주세요. "
            "특정 필자의 개성(말버릇, 독특한 표현)은 드러나지 않게 쓰세요. 설명 없이 다시 쓴 글만 출력하세요.\n\n" + text)

def _use_tag(tag):
    global OUT, NEUTRAL, GEN, RET
    if not tag: return
    OUT = ROOT / "results" / f"ko_style_{tag}"
    NEUTRAL, GEN, RET = OUT / "neutral.jsonl", OUT / "gen.jsonl", OUT / "retrieved.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if not NEUTRAL.exists() and BASE_NEUTRAL.exists():
        NEUTRAL.write_text(BASE_NEUTRAL.read_text(encoding="utf-8"), encoding="utf-8")

def read_jsonl(p):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []

def run_commit():
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts/ko_style_transfer.py", "data/style_reference.json"],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return head + (" (+미커밋 변경 있음)" if dirty else "")

def model_info():
    from src import config
    return {"model": config.GENERATION_MODEL, "backend": config.GENERATION_BACKEND, "temperature": config.GENERATION_TEMPERATURE}

def call_model(prompt, label):
    return ko_style.generate(prompt, label)

def is_429(msg):
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg

# ───────── 1) 중립문 ─────────
def select_paragraphs():
    rows = [json.loads(l) for l in (ROOT / "data/eval/eval_sources.jsonl").open(encoding="utf-8") if l.strip()]
    picked = []
    for u in PEOPLE:
        pool = sorted([r for r in rows if r["person"] == u and MIN_C <= len(r["text"]) <= MAX_C], key=lambda r: r["id"])
        random.Random(f"{SEED}-{u}").shuffle(pool)
        per_doc, chosen = Counter(), []
        for r in pool:  # 한 문서에서 2개까지 먼저, 모자라면 채움
            if per_doc[r["doc"]] < 2 and len(chosen) < PER_PERSON:
                chosen.append(r); per_doc[r["doc"]] += 1
        for r in pool:
            if len(chosen) < PER_PERSON and r not in chosen: chosen.append(r)
        if len(chosen) < PER_PERSON: sys.exit(f"[중단] {u}: {MIN_C}~{MAX_C}자 청크가 {PER_PERSON}개 미만")
        picked += chosen
    return picked

def neutralize(a):
    _use_tag(a.tag)
    OUT.mkdir(parents=True, exist_ok=True)
    sel = select_paragraphs()
    done = {r["nid"] for r in read_jsonl(NEUTRAL) if r.get("neutral")}
    commit, info = run_commit(), model_info()
    print(f"[실행 커밋] {commit} / 모델 {info['model']}")
    consec = 0
    with NEUTRAL.open("a", encoding="utf-8") as f:
        for n, r in enumerate(sel, 1):
            if r["id"] in done: continue
            rec = {"nid": r["id"], "person": r["person"], "doc": r["doc"], "type": r.get("type"), "source": r["text"],
                   "commit": commit, **info}
            try:
                rec["neutral"] = call_model(build_neutral_prompt(r["text"]), "중립화"); rec["error"] = ""; consec = 0
            except Exception as e:
                rec.update(neutral="", error=f"{type(e).__name__}: {e}"[:300])
                consec = consec + 1 if is_429(rec["error"]) else 0
            ratio = len(rec["neutral"]) / len(r["text"]) if rec["neutral"] else 0
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"[{n}/{len(sel)}] {r['id']} " + (f"ok (길이 비 {ratio:.2f})" + (" ⚠길이 차이 큼" if not 0.6 <= ratio <= 1.5 else "") if rec["neutral"] else "ERR " + rec["error"][:80]))
            if consec >= 3: sys.exit("[중단] 429 연속 3회: 쿼터 확인 후 다시 실행하면 이어서 합니다.")
    print(f"\n중립문 {len({r['nid'] for r in read_jsonl(NEUTRAL) if r.get('neutral')})}/{len(sel)}개 → {NEUTRAL.relative_to(ROOT)}")

# ───────── 2) 변환 ─────────
def heldout_docs(user):
    info = json.loads((ROOT / "data/eval/split_info.json").read_text(encoding="utf-8"))
    return {u["user"]: set(u["heldout_docs"]) for u in info["users"]}.get(user, set())

def gen(a):
    _use_tag(a.tag)
    neutrals = {r["nid"]: r for r in read_jsonl(NEUTRAL) if r.get("neutral")}
    if not neutrals:
        if not a.dry: sys.exit("[중단] 중립문이 없습니다. 먼저 neutralize를 실행하세요.")
        sel = select_paragraphs()
        neutrals = {r["id"]: {"nid": r["id"], "person": r["person"], "neutral": r["text"]} for r in sel}
        print("[dry] 중립문이 아직 없어 원문으로 대신 프롬프트만 만듭니다.\n")
    OUT.mkdir(parents=True, exist_ok=True)
    ret = json.loads(RET.read_text(encoding="utf-8")) if RET.exists() else {}
    vers = versions_of(a)
    done = {(r.get("version", "new"), r["nid"], r["target"], r["cond"]) for r in read_jsonl(GEN) if r.get("output") and not r.get("error")}
    commit, info = run_commit(), (model_info() if not a.dry else {})
    todo = len(neutrals) * len(PEOPLE) * len(CONDS) * len(vers) - len([k for k in done if k[0] in vers])
    print(f"[실행 커밋] {commit} / 남은 호출 {todo}회 (중립문 {len(neutrals)} × 목표 3 × 조건 {len(CONDS)} × 프롬프트 판 {len(vers)}: {', '.join(vers)})")
    if a.dry:
        shown = 0
        for nid, n in neutrals.items():
            for t in PEOPLE:
                for ver in vers:
                    lines, mode = p_lines_for(t, n["neutral"], ver)
                    for cond in CONDS:
                        if shown >= a.dry_n: return
                        ex = [{"text": f"(예시 본문 — {t}의 학습용 글 4개가 검색되어 들어감)"}] if "R" in cond else None
                        print(f"=== [{ver}] {nid} → {t} · {cond} · 지시 {len(lines)}줄 (판단: {mode}) ===")
                        print(build_prompt(n["neutral"], lines if "P" in cond else [], ex, ver), "\n"); shown += 1
        return
    from src.retrieve import retrieve_similar
    from src import config
    consec = 0
    with GEN.open("a", encoding="utf-8") as f:
        for nid, n in neutrals.items():
            for t in PEOPLE:
                if all((v, nid, t, c) in done for v in vers for c in CONDS): continue
                key = f"{nid}|{t}"
                if key not in ret:
                    try:
                        ex = retrieve_similar(n["neutral"], t, k=config.DEFAULT_TOP_K, exclude_doc_ids=heldout_docs(t))
                    except Exception as e:
                        sys.exit(f"[중단] 검색 실패({key}): {type(e).__name__}: {str(e)[:200]}\n학습용 색인(LINGUA_CORPUS_SPLIT=train build_index.py <user>)을 확인하세요.")
                    bad = [x["id"] for x in ex if x["id"].rsplit("_", 1)[0] in heldout_docs(t)]
                    if bad: sys.exit(f"[중단] 평가 문서가 RAG에 섞임: {bad}")
                    ret[key] = [{"id": x["id"], "score": round(float(x["score"]), 4), "type": x.get("type"), "text": x["text"]} for x in ex]
                    RET.write_text(json.dumps(ret, ensure_ascii=False, indent=1), encoding="utf-8")
                for ver, cond in [(v, c) for v in vers for c in CONDS]:
                    if (ver, nid, t, cond) in done: continue
                    rec = {"version": ver, "nid": nid, "source_person": n["person"], "target": t, "cond": cond, "commit": commit, **info}
                    try:
                        # 같은 문단·목표에는 같은 검색 결과를 두 판과 R·P+R이 공유
                        r = ko_style.convert_style(n["neutral"], t, cond, examples=ret[key], version=ver)
                        rec.update(gate_mode=r["gate_mode"], n_lines=len(r["lines"]), lines=r["lines"],
                                   example_ids=[x["id"] for x in r["examples"]], prompt=r["prompt"],
                                   output=r["converted"], error=""); consec = 0
                    except Exception as e:
                        rec.update(output="", error=f"{type(e).__name__}: {e}"[:300])
                        consec = consec + 1 if is_429(rec["error"]) else 0
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    print(f"[{ver}] {nid} → {t} {cond} " + ("ok" if rec["output"] else "ERR " + rec["error"][:80]))
                    if consec >= 3: sys.exit("[중단] 429 연속 3회: 쿼터 확인 후 다시 실행하면 이어서 합니다.")

# ───────── 3) 측정 ─────────
def _rel(p):
    try: return p.relative_to(ROOT)
    except ValueError: return p

def load_train_docs():
    info = json.loads((ROOT / "data/eval/split_info.json").read_text(encoding="utf-8"))
    return load_reference_docs({u["user"]: set(u["heldout_docs"]) for u in info["users"]})

def _classify_block(clf, kc, neutrals, allg, fmt, ver, nids):
    """한 프롬프트 판의 분류기 결과: 문단·묶음 단위, 전체 vs 형식 통과만(문단 단위만)."""
    by_src = defaultdict(list)
    for n in nids: by_src[neutrals[n]["person"]].append(n)
    for kind, title in (("para", "문단 단위(변환문 1개 = 150~450자)"), ("group", "묶음 단위(같은 원문 작성자의 문단 이어 붙임, 약 1,000자 이상)")):
        print(f"\n--- 판 {ver} · {title} ---")
        if kind == "para":
            src = [(neutrals[n]["person"], neutrals[n]["source"]) for n in nids]
            neu = [(neutrals[n]["person"], neutrals[n]["neutral"]) for n in nids]
            ev = clf.corpus["eval_paras"]
        else:
            src = [(p, "\n\n".join(neutrals[n]["source"] for n in ns)) for p, ns in by_src.items()]
            neu = [(p, "\n\n".join(neutrals[n]["neutral"] for n in ns)) for p, ns in by_src.items()]
            ev = []
        print(f"{'':<18}" + "".join(f"{m:<28}" for m in ("SVM", "Delta")) + "  (정확도 · 혼동행렬 [행=정답 강희/나경/수영, 열=판정])")
        def line(label, pairs):
            if not pairs: print(f"{label:<18}(해당 없음)"); return
            truth = [t for t, _ in pairs]; pr = clf.predict([x for _, x in pairs])
            cells = [f"{kc.accuracy(truth, pr[m]):.2f} {kc.confusion(truth, pr[m])}" for m in ("SVM", "Delta")]
            print(f"{label:<18}" + "  ".join(f"{c:<28}" for c in cells) + f"  n={len(pairs)}")
        line("상한: 원문", src)
        if ev: line("상한: 평가문서 전체", ev)
        line("N 중립(작성자)", neu)
        npred = clf.predict([x for _, x in neu])
        print(f"{'N 판정 분포':<18}" + "  ".join(f"{m} " + str(dict(Counter(npred[m]))) for m in ("SVM", "Delta")) + "  ← 한 사람으로 몰리는지 먼저 볼 것(D-047)")
        for cond in CONDS:
            if kind == "para":
                allp = [(t, allg[(ver, n, t, cond)]["output"], fmt[(ver, n, t, cond)]["pass"]) for t in PEOPLE for n in nids]
            else:
                allp = [(t, "\n\n".join(allg[(ver, n, t, cond)]["output"] for n in ns), all(fmt[(ver, n, t, cond)]["pass"] for n in ns))
                        for t in PEOPLE for ns in by_src.values()]
            line(f"{cond} 전체", [(t, x) for t, x, _ in allp])
            line(f"{cond} 형식통과만", [(t, x) for t, x, ok in allp if ok])  # 묶음은 묶음 안 문단이 모두 통과한 것만

def report(a):
    import numpy as np
    _use_tag(a.tag)
    if not kiwi_ok(): sys.exit("[중단] kiwipiepy가 필요합니다(Mac의 .venv에서 실행).")
    neutrals = {r["nid"]: r for r in read_jsonl(NEUTRAL) if r.get("neutral")}
    allg = {}
    for r in read_jsonl(GEN):
        if r.get("output") and not r.get("error"): allg[(r.get("version", "new"), r["nid"], r["target"], r["cond"])] = r
    def complete(v): return [n for n in neutrals if all((v, n, t, c) in allg for t in PEOPLE for c in CONDS)]
    vers_present = [v for v in PROMPT_VERSIONS if complete(v)]
    if not vers_present: sys.exit("[중단] 완성된 세트가 없습니다(프롬프트 판별로 목표 3 × 조건 3 모두 필요).")
    for v in PROMPT_VERSIONS: print(f"프롬프트 판 {v}: 완성된 중립문 {len(complete(v))}/{len(neutrals)}개")
    ver0 = a.version if a.version in vers_present else vers_present[0]
    print(f"①② 는 '{ver0}' 판 기준. ④⑤ 는 있는 판 모두.")
    gens = {(n, t, c): r for (v, n, t, c), r in allg.items() if v == ver0}
    nids = complete(ver0)
    print("실행 커밋: " + ", ".join(sorted({r['commit'] for r in allg.values()})) + " / 모델: " + ", ".join(sorted({r.get('model', '?') for r in allg.values()})) + "\n")
    from src import ko_format
    fmt = {k: ko_format.check(neutrals[k[1]]["neutral"], r["output"]) for k, r in allg.items()}  # 형식 검사: 규칙, 변환 입력(중립문) 대비
    docs = load_train_docs()
    print(f"기준 프로필: 학습용 문서 {len(docs)}편 (평가 문서 제외), 기능 형태소 top100, Burrows' Delta\n")
    short = {"ganghee": "강희", "nakyung": "나경", "suyoung": "수영"}

    # ① 3×3 귀속: (목표 × 조건)별 출력을 이어 붙여 한 단위로 판정
    for exclude in (False, True):
        ref = build_ref(docs, exclude)
        print(f"== ① 작성자 귀속 ({'지시 그룹 표현 뺀 버전' if exclude else '지시 그룹 표현 넣은 버전'}) | 값 = 각 사람 기준과의 Delta 거리(작을수록 가까움) ==")
        print("단위(목표·조건)      | 강희    나경    수영   | 판정   | 목표쪽 여유(다른 둘 평균 − 목표)")
        stat = defaultdict(list)
        units = []
        for cond in CONDS:
            for t in PEOPLE:
                units.append((f"{short[t]} {cond}", t, cond, " ".join(gens[(n, t, cond)]["output"] for n in nids)))
        for t in PEOPLE:
            units.append((f"{short[t]} N(중립)", t, "N", " ".join(neutrals[n]["neutral"] for n in nids)))  # N은 목표와 무관하나 여유 계산은 목표별
        for label, t, cond, text in units:
            z, ntok = zvec(ref, text, exclude)
            d = {p: float(np.abs(z - ref["cents"][p]).mean()) for p in PEOPLE}
            pred = min(d, key=d.get)
            margin = float(np.mean([d[p] for p in PEOPLE if p != t]) - d[t])
            stat[cond].append((pred == t, margin))
            print(f"{label:<14} {ntok:>4}토큰 | {d['ganghee']:.3f}  {d['nakyung']:.3f}  {d['suyoung']:.3f} | {short[pred]}{'(맞음)' if pred == t else ''}   | {margin:+.3f}")
        mn = [m for _, m in stat["N"]]  # 목표 순서(PEOPLE)대로 쌓임
        print("  조건별 요약: " + " / ".join(
            f"{c} 맞힘 {sum(x for x, _ in v)}/3 · 여유 평균 {np.mean([m for _, m in v]):+.3f} · N 대비 {np.mean([m - b for (_, m), b in zip(v, mn)]):+.3f}"
            for c, v in stat.items() if c != "N") + f" / N(중립) 여유 평균 {np.mean(mn):+.3f}")
        print("  읽는 법: 글 내용(12개 문단)이 같아 N도 여유가 0이 아님 → '여유'의 절대값이 아니라 'N 대비' 변화가 + 인지를 본다. N의 맞힘은 목표와 무관해 참고용.\n")

    # ② 본인 문단의 원문 거리(중립문 대비)
    for exclude in (False, True):
        ref = build_ref(docs, exclude)
        print(f"== ② 원문과의 거리 ({'뺀 버전' if exclude else '넣은 버전'}) | 문단 단위 Delta 거리, 작을수록 원문에 가까움 ==")
        rows = defaultdict(list)
        for n in nids:
            s = neutrals[n]["person"]
            zo, _ = zvec(ref, neutrals[n]["source"], exclude)
            dist = lambda txt: float(np.abs(zo - zvec(ref, txt, exclude)[0]).mean())
            base = dist(neutrals[n]["neutral"]); rows[("N", s)].append(base)
            for cond in CONDS:
                rows[(cond, "own")].append(dist(gens[(n, s, cond)]["output"]) - base)
                rows[(cond, "other")].extend(dist(gens[(n, t, cond)]["output"]) - base for t in PEOPLE if t != s)
        n_own = len(nids)
        print(f"중립문→원문 거리 평균 {np.mean([v for (c, _), vs in rows.items() if c == 'N' for v in vs]):.3f} (문단 {n_own}개)")
        print("조건 | 본인 문체로 변환: 거리 변화(변환−중립, 음수=원문에 가까워짐) 평균 · 가까워진 문단 | 다른 사람 문체로 변환(대조): 평균 · 가까워진 비율")
        for cond in CONDS:
            own, oth = rows[(cond, "own")], rows[(cond, "other")]
            print(f"{cond:<4} | {np.mean(own):+.3f} · {sum(x < 0 for x in own)}/{len(own)} | {np.mean(oth):+.3f} · {sum(x < 0 for x in oth)}/{len(oth)}")
        print("  해석: 본인 쪽이 대조(다른 사람 쪽)보다 더 음수일 때만 '그 사람 문체로 옮겨졌다'는 신호. 문단 하나는 토큰이 적어 값이 불안정함.\n")

    # ③ 사람용 CSV (두 판 모두 한 파일, '판' 열로 구분. 형식 검사는 규칙 결과이며 의미·문체 판단이 아님)
    pairs = []
    for ver in vers_present:
        for n in complete(ver):
            for t in PEOPLE:
                for cond in CONDS:
                    g = allg[(ver, n, t, cond)]; fc = fmt[(ver, n, t, cond)]
                    pairs.append({"판": ver, "문단": n, "원문 작성자": short[neutrals[n]["person"]], "목표": short[t], "조건": cond, "지시 줄 수": g["n_lines"],
                                  "지시": " / ".join(g["lines"]), "원문(정답)": neutrals[n]["source"], "중립문": neutrals[n]["neutral"],
                                  "변환문": g["output"], "형식 검사": "통과" if fc["pass"] else "표시", "형식 사유": "; ".join(fc["reasons"]),
                                  "문장수 중립→변환": f"{fc['src']['sentences']}→{fc['out']['sentences']}",
                                  "의미 유지(O/X)": "", "메모": ""})
    p1 = OUT / "ko_style_pairs.csv"
    with p1.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys())); w.writeheader(); w.writerows(pairs)
    rng = random.Random(SEED)
    blind = rng.sample(pairs, min(a.blind_n, len(pairs)))
    p2, p3 = OUT / "ko_style_blind.csv", OUT / "ko_style_blind_key.csv"
    with p2.open("w", newline="", encoding="utf-8-sig") as f, p3.open("w", newline="", encoding="utf-8-sig") as g2:
        w = csv.writer(f); w.writerow(["번호", "글", "누구 글 같은가(강희/나경/수영/모름)", "확신(1~3)"])
        k = csv.writer(g2); k.writerow(["번호", "판", "문단", "원문 작성자", "목표", "조건"])
        for i, r in enumerate(blind, 1):
            w.writerow([i, r["변환문"], "", ""]); k.writerow([i, r["판"], r["문단"], r["원문 작성자"], r["목표"], r["조건"]])
    print(f"사람 읽기 CSV → {_rel(p1)} (의미 유지 O/X는 사람이 표시, 형식 검사 열은 규칙 결과)")
    print(f"블라인드 시트 {len(blind)}행 → {_rel(p2)} (정답표는 {p3.name}, 판단이 끝나기 전에는 열지 않기)")

    # ⑤ 형식 검사(D-063): 규칙, 사후 계산. 통과 = 표시 사유 없음
    print("\n== ⑤ 형식 검사 통과율 (규칙: 문단 수·목록 줄·명사형 종결·문장 수·글자 수 비) ==")
    for ver in vers_present:
        ns = complete(ver)
        print(f"-- 판 {ver} (중립문 {len(ns)}개) --  통과/전체 | 목표 강희·나경·수영")
        for cond in CONDS:
            per = [sum(fmt[(ver, n, t, cond)]["pass"] for n in ns) for t in PEOPLE]
            print(f"{cond:<4} {sum(per)}/{len(ns) * 3}  | " + "  ".join(f"{short[t]} {x}/{len(ns)}" for t, x in zip(PEOPLE, per)))
        why = Counter(r.split(" ")[0] + (" " + r.split(" ")[1] if r.startswith("글자") else "") for (v, *_), fc in fmt.items() if v == ver for r in fc["reasons"])
        print("  표시 사유 분포: " + (", ".join(f"{k} {c}" for k, c in why.most_common()) or "없음"))
        for (v, n, t, c), fc in sorted(fmt.items()):
            if v == ver and not fc["pass"]: print(f"  표시 {n} → {short[t]} {c}: {'; '.join(fc['reasons'])}")
    print("  주의: 임계값은 초안(선택용 12문단에서 조정, 확인용에서는 고정). 수영 줄은 이웃 문장 잇기를 허용하므로 문장 수 감소 1은 통과, 2 이상만 표시. 통과 ≠ 좋은 변환(의미·문체는 사람이 본다).\n")
    # ④ 분류기 두 개(D-053): 선형 SVM(문단 학습) + Delta(문서 학습), 지시 그룹 표현 뺀 기능 형태소. 판별·전체/형식 통과만
    from src import ko_classify as kc
    clf = kc.Classifiers()
    print("== ④ 분류기 두 개 (SVM·Delta 같은 특징: 기능 형태소, 지시 그룹 표현 뺌) ==")
    print(clf.sizes())
    for ver in vers_present:
        _classify_block(clf, kc, neutrals, allg, fmt, ver, complete(ver))
    print("\n읽는 법: 정답 = 목표 인물. 우연 0.33. 상한(원문)과 N 사이 어디에 오는지로 읽는다. 두 분류기가 같은 방향이면 견고, 갈리면 상한이 높은 SVM을 주 결과로.")
    print("주의: 조건당 문단 36개·묶음 9개는 구간이 넓다(±15%p 이상). 묶음은 같은 문단이 겹쳐 독립이 아니다. 형식 통과만 보면 표본이 더 줄어든다. 분류기는 사람 글로만 학습해 AI가 쓴 글이 한 사람으로 몰릴 수 있다. 의미 유지는 사람이 본다.")

    print("\n주의: 중립문 12개·단위당 3개뿐이라 방향만 봅니다. 지시에 쓴 표현은 '뺀 버전' 결과를 우선 보세요.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("neutralize", "gen", "report"):
        p = sp.add_parser(name); p.add_argument("--tag", default=None, help="모델별 결과 폴더 이름(results/ko_style_<tag>)")
        if name == "gen":
            p.add_argument("--dry", action="store_true", help="API 없이 프롬프트만 출력"); p.add_argument("--dry-n", type=int, default=6)
            p.add_argument("--version", choices=["old", "new", "both"], default="both", help="프롬프트 판(old=커밋 24ae462, new=현재). 기본 both")
        if name == "report":
            p.add_argument("--blind-n", type=int, default=36)
            p.add_argument("--version", choices=list(PROMPT_VERSIONS), default="new", help="①②에 쓸 판(③④⑤는 있는 판 모두)")
    a = ap.parse_args(); {"neutralize": neutralize, "gen": gen, "report": report}[a.cmd](a)
