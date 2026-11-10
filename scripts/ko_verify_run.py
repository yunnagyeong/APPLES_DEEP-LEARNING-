"""한→한 문체 변환 검증 파이프라인 (강희 담당). API·LLM 없음, Kiwi + 규칙(src/ko_verify.py).

생성(짝 예시 검색 + Gemini 변환)과 분류·정답률은 팀원 코드가 맡는다. 이 스크립트는 그 사이에서
"이 변환이 의미를 지켰는가 / 평가가 오염되지 않았는가"만 판정해 파일로 넘긴다.

  gen   : 변환 결과 검증 → results/ko_verify/<tag>/verify.jsonl, verify.csv, summary.md
          python scripts/ko_verify_run.py gen --gen <변환결과.jsonl> [--tag 이름] [--override 확인.csv]
  pairs : 짝 예시 DB 검증(AI 글이 사람 글 뜻을 바꿨나, 평가 문서가 섞였나) → results/ko_verify/pairs_<user>.csv
          python scripts/ko_verify_run.py pairs --pairs <짝.jsonl> --user ganghee
  demo  : 기존 결과(results/ko_doc/base)를 gen 형식으로 바꿔 한 번 돌려 본다.

팀원에게 넘기는 것: verify.jsonl 한 줄 = {"id", "verdict", "use", "leak", "issues", "stats"}
  verdict : PASS | REVIEW | FAIL  (사람이 --override로 확정한 것은 그 값)
  use     : 정답률 계산에 넣을지. 유출이 있으면 false(지표에서 뺌)
  엄격 정답률 = (분류 맞힘 AND verdict == PASS) / (use == true 인 건수)
입력 형식과 판정 기준: docs/KO_VERIFY.md
"""
import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import ko_verify as kv  # noqa: E402

PEOPLE = ["ganghee", "nakyung", "suyoung"]
LEAK_OVERLAP = 0.5  # 글자 5-gram 기준. 예시가 정답(사람 원문)이나 입력을 이만큼 덮으면 유출
OUT = ROOT / "results" / "ko_verify"


# ---------------------------------------------------------------- 공통
def read_jsonl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def heldout_docs() -> dict[str, set[str]]:
    info = json.loads((ROOT / "data/eval/split_info.json").read_text(encoding="utf-8"))
    return {u["user"]: set(u["heldout_docs"]) for u in info["users"]}


def doc_of(example_id: str) -> str:
    """'doc_007_08' → 'doc_007' (청크 번호 떼기). 이미 문서 id면 그대로."""
    return example_id.rsplit("_", 1)[0] if re.fullmatch(r".+_\d+_\d+", example_id or "") else example_id


def _grams(t: str, n: int = 5) -> set[str]:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def containment(part: str, whole: str) -> float:
    """part의 글자 5-gram 중 whole에 있는 비율."""
    g = _grams(part)
    return len(g & _grams(whole)) / len(g) if g else 0.0


def short_issues(v: dict) -> str:
    return kv.summarize(v).split(" | ", 1)[-1] if v["issues"] else ""


# ---------------------------------------------------------------- 유출 검사
def leak_reasons(rec: dict, held: dict[str, set[str]]) -> list[str]:
    """평가가 오염됐는지: 평가 문서가 예시로 들어감 / 정답(사람 원문)이 예시에 들어 있음 / 입력과 같은 글이 예시."""
    why = []
    exs = [e for e in rec.get("examples") or [] if isinstance(e, dict)]
    for e in exs:
        eid = str(e.get("id", ""))
        if eid and doc_of(eid) in held.get(rec["target"], set()):
            why.append(f"예시 {eid}가 평가 문서")
        styled = e.get("styled") or e.get("text") or ""
        if rec.get("source") and styled and containment(rec["source"], styled) >= LEAK_OVERLAP:
            why.append(f"예시 {eid or '?'}에 정답(사람 원문)이 들어 있음")
        if e.get("ai") and containment(rec["input"], e["ai"]) >= LEAK_OVERLAP:
            why.append(f"예시 {eid or '?'}의 AI 글이 입력과 같은 글")
        if rec.get("source_doc") and rec.get("source_person") == rec["target"] and doc_of(eid) == rec["source_doc"]:
            why.append(f"예시 {eid}가 입력과 같은 문서")
    return sorted(set(why))


# ---------------------------------------------------------------- gen
def validate(recs):
    bad = [i for i, r in enumerate(recs)
           if not all(k in r for k in ("id", "target", "input", "output")) or r["target"] not in PEOPLE]
    if bad:
        sys.exit(f"[중단] 필수 필드(id, target, input, output) 누락이나 알 수 없는 target: {len(bad)}건 (첫 줄 {bad[0] + 1})")
    dup = [k for k, c in Counter(r["id"] for r in recs).items() if c > 1]
    if dup:
        sys.exit(f"[중단] id 중복: {dup[:5]}")


def load_override(path):
    if not path:
        return {}
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = {r["id"]: r["사람판정"].strip().upper() for r in rows
           if r.get("사람판정", "").strip().upper() in ("PASS", "FAIL")}
    print(f"사람 확인으로 덮어쓴 판정 {len(out)}건 ({path})")
    return out


def run_gen(recs, out_dir: Path, override: dict):
    validate(recs)
    held = heldout_docs()
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in recs:
        r.setdefault("cond", "-")
        r["leak"] = leak_reasons(r, held)
        r["v"] = kv.verify(r["input"], r["output"], r.get("examples"))
        r["verdict"] = override.get(r["id"], r["v"]["verdict"])

    with (out_dir / "verify.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({"id": r["id"], "cond": r["cond"], "target": r["target"], "verdict": r["verdict"],
                                "auto_verdict": r["v"]["verdict"], "use": not r["leak"], "leak": r["leak"],
                                "issues": r["v"]["issues"], "stats": r["v"]["stats"]}, ensure_ascii=False) + "\n")
    with (out_dir / "verify.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "조건", "목표", "자동판정", "사람판정", "유출", "문제", "근거 문장",
                    "내용 재현율", "문체 변화율", "거의 그대로", "입력", "출력"])
        for r in recs:
            ev = next((i["evidence"] for i in r["v"]["issues"] if i["severity"] != "minor" and i["evidence"]), "")
            st = r["v"]["stats"]
            w.writerow([r["id"], r["cond"], r["target"], r["v"]["verdict"], override.get(r["id"], ""),
                        "; ".join(r["leak"]), short_issues(r["v"]), ev, st.get("content_recall"),
                        st.get("style_change"), "예" if st.get("unchanged") else "", r["input"], r["output"]])

    L = [f"# 한→한 변환 검증 결과 ({len(recs)}건)\n"]
    leaks = [r for r in recs if r["leak"]]
    if leaks:
        L.append(f"**유출 {len(leaks)}건 → use=false (정답률에서 뺄 것).** 예: {leaks[0]['id']} — {leaks[0]['leak'][0]}\n")
    L += ["| 조건 | n | PASS | REVIEW | FAIL | 유출 | 거의 그대로 |", "|---|---|---|---|---|---|---|"]
    for cond in sorted({r["cond"] for r in recs}):
        g = [r for r in recs if r["cond"] == cond]
        c = Counter(r["verdict"] for r in g)
        L.append(f"| {cond} | {len(g)} | {c['PASS']} | {c['REVIEW']} | {c['FAIL']} | {sum(bool(r['leak']) for r in g)} | "
                 f"{sum(bool(r['v']['stats'].get('unchanged')) for r in g)} |")
    why = Counter(i["label_ko"] for r in recs for i in r["v"]["issues"] if i["severity"] in ("major", "critical"))
    L.append("\n걸린 사유(major·critical): " + (", ".join(f"{k} {v}" for k, v in why.most_common()) or "없음"))
    L.append(f"\n사람 확인으로 덮어쓴 판정 {sum(r['id'] in override for r in recs)}건. REVIEW는 verify.csv의 "
             "'사람판정' 열에 PASS/FAIL을 적고 --override로 다시 돌리면 반영된다.")
    L.append("'거의 그대로'(어절 97% 이상 같음)가 많으면 변환이 안 된 것이라 분류 정답률이 입력 판정과 같아진다.")
    text = "\n".join(L)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n→ {out_dir.relative_to(ROOT)}/verify.jsonl (팀원 전달용), verify.csv (사람 확인용), summary.md")


def cmd_gen(a):
    run_gen(read_jsonl(a.gen), OUT / (a.tag or Path(a.gen).stem), load_override(a.override))


# ---------------------------------------------------------------- pairs
def cmd_pairs(a):
    """짝 = (사람 글 styled, 그 글을 AI 말투로 바꾼 ai). AI 글이 뜻을 바꿨으면 모델이 '뜻 바꾸기'를 배운다."""
    pairs = read_jsonl(a.pairs)
    held = heldout_docs().get(a.user, set())
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"pairs_{a.user}.csv"
    vc, rows = Counter(), []
    for p in pairs:
        if not p.get("ai") or not p.get("styled"):
            sys.exit(f"[중단] 짝에 ai·styled가 모두 있어야 합니다: {p.get('id')}")
        v = kv.verify(p["styled"], p["ai"])  # 사람 글(원본) → AI 글(바꾼 것)
        is_held = doc_of(str(p.get("id", ""))) in held
        verdict = "FAIL" if is_held else v["verdict"]
        vc[verdict] += 1
        rows.append([p.get("id", ""), verdict, "평가 문서" if is_held else "", short_issues(v), p["styled"], p["ai"]])
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "판정", "유출", "문제", "사람 글", "AI 글"])
        w.writerows(rows)
    print(f"짝 {len(pairs)}개: PASS {vc['PASS']} · REVIEW {vc['REVIEW']} · FAIL {vc['FAIL']} → {out.relative_to(ROOT)}")
    print("FAIL(평가 문서 포함)은 반드시 빼고, REVIEW는 읽어 보고 빼거나 AI 글을 다시 만든다.")


# ---------------------------------------------------------------- demo
def cmd_demo(a):
    ret = json.loads((ROOT / "results/ko_doc/base/retrieved.json").read_text(encoding="utf-8"))
    recs = []
    for r in read_jsonl(ROOT / "results/ko_doc/base/gen.jsonl"):
        if not r.get("output") or r.get("error"):
            continue
        key = next((k for k in ret if k.startswith(f"{r['set']}|{r['doc']}|{r['target']}|")), None)
        exs = [{"id": e["id"], "styled": e["text"]} for e in ret.get(key, [])] if "R" in r["cond"] else []
        recs.append({"id": f"{r['doc']}|{r['target']}|{r['cond']}", "cond": r["cond"], "target": r["target"],
                     "input": (ROOT / "data/ko_docs" / r["set"] / f"{r['doc']}.txt").read_text(encoding="utf-8").strip(),
                     "output": r["output"], "examples": exs})
    print(f"기존 결과 {len(recs)}건을 gen 형식으로 바꿔 검증합니다(짝 예시 아님, 동작 확인용).\n")
    run_gen(recs, OUT / "demo_ko_doc_base", {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("gen")
    p.add_argument("--gen", required=True, help="변환 결과 JSONL (형식: docs/KO_VERIFY.md)")
    p.add_argument("--tag", default=None, help="결과 폴더 이름(results/ko_verify/<tag>), 기본은 파일 이름")
    p.add_argument("--override", default=None, help="사람 확인 CSV(id, 사람판정=PASS|FAIL)")
    p = sp.add_parser("pairs")
    p.add_argument("--pairs", required=True, help="짝 예시 JSONL: {id, ai, styled}")
    p.add_argument("--user", required=True, choices=PEOPLE)
    sp.add_parser("demo")
    a = ap.parse_args()
    {"gen": cmd_gen, "pairs": cmd_pairs, "demo": cmd_demo}[a.cmd](a)
