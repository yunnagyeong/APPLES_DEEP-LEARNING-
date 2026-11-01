"""의미 검증 v2(src/verify.py) 평가.

두 가지 평가셋
  1) error_cases.csv (기본) — 사람이 심각도를 단 32문장. v2 기준을 이걸 보고 정했으므로 낙관적인 숫자.
  2) data/measure/verifier_heldout.csv (--heldout) — 새로 만든 평가셋 64건. error_cases와 겹치지 않는 원문
     (짧은 문장 8 + 팀 코퍼스 긴 문단 8)마다 [정확한 번역 / 문체를 크게 바꾼 번역 / 오류를 하나 넣은 번역 2개].
     넣은 오류 유형과 기대 판정을 알고 있으므로 사람 라벨 없이 채점됨. 긴 문단은 오류를 한 문장에만 숨김.

사용법 (프로젝트 루트):
  python scripts/eval_verifier.py --heldout              # 새 평가셋 64건 (4건씩 동시, 약 2~3분)
  python scripts/eval_verifier.py --heldout --repeat 2   # 같은 입력을 2번씩 → 판정이 매번 같은지(안정성)도 봄
  python scripts/eval_verifier.py                        # 기존 32건
  python scripts/eval_verifier.py --heldout --report     # API 호출 없이 저장된 결과로 요약만
  python scripts/eval_verifier.py --model gemini-3.5-flash-lite --heldout --out data/measure/heldout_lite.jsonl

- 사례당 판정 모델 1회 + 임베딩 1회. 결과는 끝나는 대로 한 줄씩 저장되어, 중간에 끊겨도 다시 실행하면 이어서 함.
- 보고: 오탐(정상·문체변경 번역을 REVIEW/FAIL로), 오류 탐지율, 판정 일치(기대 PASS/REVIEW/FAIL과 같은지),
  유형 일치(판정 모델이 넣은 오류 유형을 맞게 불렀는지), 오류 유형별·짧은 문장/긴 문단별 결과, 반복 안정성, 걸린 시간.
"""

import argparse
import csv
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEV_ORDER = ["정상", "하", "중", "상"]
HELDOUT_CASES = ROOT / "data/measure/verifier_heldout.csv"
HELDOUT_OUT = ROOT / "data/measure/verifier_v2_heldout.jsonl"


def load_runs(path: Path) -> dict:
    """{(id, run): record}. run이 없는 예전 기록은 run=1."""
    if not path.exists():
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[(str(r["id"]), int(r.get("run", 1)))] = r
    return out


def load_jsonl(path):
    if not path.exists():
        return {}
    return {str(json.loads(l)["id"]): json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def auc(normal, errors):
    """오류 사례 점수가 정상 사례보다 낮은 쌍의 비율(동점 0.5)."""
    if not normal or not errors:
        return None
    return sum((e < n) + 0.5 * (e == n) for n in normal for e in errors) / (len(normal) * len(errors))


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def _issues_str(r):
    return ", ".join(f"{i['severity'][:3]}:{i['type']}" for i in r["issues"]) or "-"


# ---------------------------------------------------------------- 기존 32건(사람 심각도) 보고
def report_human(cases, first):
    rows = [(c, first[c["번호"]]) for c in cases if c["번호"] in first]
    sev = lambda c: c["심각도"] or "정상"  # noqa: E731
    errs = [(c, r) for c, r in rows if sev(c) != "정상"]
    oks = [(c, r) for c, r in rows if sev(c) == "정상"]

    print(f"\n=== v2 검증 결과 ({len(rows)}/{len(cases)}건, 오류 {len(errs)} · 정상 {len(oks)}) ===")
    print(f"{'#':>3} {'사람':4} {'판정':6} {'점수':>5}  문제")
    for c, r in rows:
        print(f"{c['번호']:>3} {sev(c):4} {r['verdict']:6} {r['meaning_score']:5.0f}  {_issues_str(r)}")

    flagged = lambda r: r["verdict"] != "PASS"  # noqa: E731
    any_issue = lambda r: bool(r["issues"])  # noqa: E731
    print("\n[v2] 오류 탐지(REVIEW·FAIL):", f"{sum(flagged(r) for _, r in errs)}/{len(errs)}",
          "| 정상 오탐:", f"{sum(flagged(r) for _, r in oks)}/{len(oks)}")
    print("[v2] minor까지 포함한 지적:", f"{sum(any_issue(r) for _, r in errs)}/{len(errs)}",
          "| 정상에 지적:", f"{sum(any_issue(r) for _, r in oks)}/{len(oks)}")
    for s in SEV_ORDER:
        sc = [r["meaning_score"] for c, r in rows if sev(c) == s]
        if sc:
            vd = {v: sum(1 for c, r in rows if sev(c) == s and r["verdict"] == v) for v in ("PASS", "REVIEW", "FAIL")}
            print(f"  {s:3}: n={len(sc):2d} 평균 점수 {sum(sc)/len(sc):5.1f}  판정 {vd}")
    a = auc([r["meaning_score"] for _, r in oks], [r["meaning_score"] for _, r in errs])
    if a is not None:
        print(f"[v2] 점수 분리도(AUC, 오류<정상): {a:.3f}")

    legacy = load_jsonl(ROOT / "data/measure/meaning_evaluation.jsonl")
    nuance = load_jsonl(ROOT / "data/measure/nuance_evaluation.jsonl")
    ids = [c["번호"] for c, _ in rows if c["번호"] in legacy and c["번호"] in nuance]
    if ids:
        by = {c["번호"]: c for c, _ in rows}
        le = [i for i in ids if sev(by[i]) != "정상"]
        lo = [i for i in ids if sev(by[i]) == "정상"]
        lf = lambda i: legacy[i]["verification"]["verdict"] != "PASS" or nuance[i]["nuance"]["has_meaning_error"]  # noqa: E731
        print(f"[레거시 NLI+뉘앙스, 같은 {len(ids)}건] 오류 탐지 {sum(map(lf, le))}/{len(le)} | 정상 오탐 {sum(map(lf, lo))}/{len(lo)}")


# ---------------------------------------------------------------- 새 평가셋(기대 판정) 보고
def _detected(c, r):
    """기대가 REVIEW/FAIL이면 PASS가 아니어야, 기대가 PASS(minor 오류)면 문제를 하나라도 지적해야 탐지."""
    if c["기대판정"] in ("REVIEW", "FAIL"):
        return r["verdict"] != "PASS"
    return bool(r["issues"])


def _type_hit(c, r):
    ok = set(filter(None, c["허용유형"].split("|")))
    return bool(ok & {i["type"] for i in r["issues"]})


def report_heldout(cases, first):
    rows = [(c, first[c["번호"]]) for c in cases if c["번호"] in first]
    normals = [(c, r) for c, r in rows if c["변형"] in ("정확", "문체변경")]
    errors = [(c, r) for c, r in rows if c["변형"] == "오류"]
    major = [(c, r) for c, r in errors if c["기대판정"] in ("REVIEW", "FAIL")]
    minor = [(c, r) for c, r in errors if c["기대판정"] == "PASS"]

    print(f"\n=== 새 평가셋 결과 ({len(rows)}/{len(cases)}건) ===")
    print(f"{'번호':7} {'변형':5} {'넣은 오류':20} {'기대':6} {'판정':6} {'점수':>5}  찾은 문제")
    for c, r in rows:
        mark = ""
        if c["변형"] != "오류" and r["verdict"] != "PASS":
            mark = "  ← 오탐"
        elif c["변형"] == "오류" and not _detected(c, r):
            mark = "  ← 놓침"
        elif c["변형"] == "오류" and r["verdict"] != c["기대판정"]:
            mark = "  ← 심각도 다름"
        print(f"{c['번호']:7} {c['변형']:5} {c['주입오류'] or '-':20} {c['기대판정']:6} {r['verdict']:6} "
              f"{r['meaning_score']:5.0f}  {_issues_str(r)}{mark}")

    fp = lambda xs: sum(r["verdict"] != "PASS" for _, r in xs)  # noqa: E731
    print("\n[오탐] 정상 번역을 REVIEW·FAIL로 판정")
    for name in ("정확", "문체변경"):
        xs = [(c, r) for c, r in normals if c["변형"] == name]
        print(f"  {name:5}: {pct(fp(xs), len(xs))}")
    print(f"  전체 : {pct(fp(normals), len(normals))}")

    print("\n[탐지] 넣은 오류를 잡았나")
    print(f"  major·critical 오류 → REVIEW·FAIL : {pct(sum(_detected(c, r) for c, r in major), len(major))}")
    print(f"  minor 오류 → 문제로 지적          : {pct(sum(_detected(c, r) for c, r in minor), len(minor))}")
    print(f"  판정까지 기대와 같음               : {pct(sum(r['verdict'] == c['기대판정'] for c, r in errors), len(errors))}")
    print(f"  오류 유형까지 맞게 지목            : {pct(sum(_type_hit(c, r) for c, r in errors), len(errors))}")

    print("\n[짧은 문장 vs 긴 문단]")
    for kind in ("짧은 문장", "긴 문단"):
        e = [(c, r) for c, r in errors if c["원문종류"] == kind]
        n = [(c, r) for c, r in normals if c["원문종류"] == kind]
        secs = [r["seconds"] for c, r in rows if c["원문종류"] == kind and r.get("seconds")]
        t = f" | 검증 1회 평균 {st.mean(secs):.1f}초 (최대 {max(secs):.1f})" if secs else ""
        print(f"  {kind}: 탐지 {pct(sum(_detected(c, r) for c, r in e), len(e))} | 오탐 {pct(fp(n), len(n))}{t}")

    print("\n[오류 유형별 탐지]")
    by = defaultdict(list)
    for c, r in errors:
        by[c["주입오류"]].append((c, r))
    for typ, xs in sorted(by.items(), key=lambda kv: kv[0]):
        print(f"  {typ:22} {pct(sum(_detected(c, r) for c, r in xs), len(xs)):12} "
              f"판정 일치 {sum(r['verdict'] == c['기대판정'] for c, r in xs)}/{len(xs)}")

    a = auc([r["meaning_score"] for _, r in normals], [r["meaning_score"] for _, r in major])
    if a is not None:
        print(f"\n[점수 분리도] major·critical 오류 < 정상 (AUC): {a:.3f}")

    misses = [(c, r) for c, r in rows if (c["변형"] != "오류" and r["verdict"] != "PASS")
              or (c["변형"] == "오류" and not _detected(c, r))]
    if misses:
        print("\n[확인할 사례] 오탐·놓친 것 — 판정 모델의 설명")
        for c, r in misses:
            print(f"  {c['번호']} ({c['변형']}, 넣은 오류 {c['주입오류'] or '없음'}: {c['설명']})")
            for i in r["issues"]:
                print(f"     - [{i['severity']}] {i['type']}: {i.get('note', '')}  “{i.get('evidence', '')[:80]}”")
            if not r["issues"]:
                print("     - 찾은 문제 없음")


def report_stability(runs):
    by = defaultdict(dict)
    for (cid, run), r in runs.items():
        by[cid][run] = r
    multi = {cid: rs for cid, rs in by.items() if len(rs) > 1}
    if not multi:
        return
    same = sum(len({r["verdict"] for r in rs.values()}) == 1 for rs in multi.values())
    spread = [max(r["meaning_score"] for r in rs.values()) - min(r["meaning_score"] for r in rs.values())
              for rs in multi.values()]
    print(f"\n[안정성] 같은 입력을 여러 번 검증했을 때 판정이 모두 같음: {pct(same, len(multi))}"
          f" | 점수 차이 평균 {st.mean(spread):.1f}점 (최대 {max(spread):.1f})")
    diff = [cid for cid, rs in multi.items() if len({r['verdict'] for r in rs.values()}) > 1]
    if diff:
        print("  판정이 바뀐 사례:", ", ".join(
            f"{cid}({'/'.join(rs[k]['verdict'] for k in sorted(rs))})" for cid, rs in multi.items() if cid in diff))


def report_time(runs):
    secs = [r["seconds"] for r in runs.values() if r.get("seconds")]
    if secs:
        print(f"\n[시간] 검증 1회: 평균 {st.mean(secs):.1f}초 · 중앙값 {st.median(secs):.1f}초 · 최대 {max(secs):.1f}초 (n={len(secs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", action="store_true", help="새 평가셋(data/measure/verifier_heldout.csv)")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeat", type=int, default=1, help="같은 입력을 몇 번씩 검증할지(안정성 확인)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4, help="동시에 평가할 사례 수")
    ap.add_argument("--model", default=None, help="판정 모델 바꾸기(기본 config.VERIFY_MODEL)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    cases_path = Path(a.cases) if a.cases else (HELDOUT_CASES if a.heldout else ROOT / "error_cases.csv")
    out = Path(a.out) if a.out else (HELDOUT_OUT if a.heldout else ROOT / "data/measure/verifier_v2_eval.jsonl")
    cases = list(csv.DictReader(open(cases_path, encoding="utf-8-sig")))
    runs = load_runs(out)

    if not a.report:
        from src import config
        from src.verify import verify_translation
        if a.model:
            config.VERIFY_MODEL = a.model
        todo = [(c, k) for k in range(1, a.repeat + 1) for c in cases if (c["번호"], k) not in runs][: a.limit]
        print(f"평가 {len(todo)}건 (이미 {len(runs)}건 완료) → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        lock = threading.Lock()

        def _one(c, k):
            t0 = time.perf_counter()
            v = verify_translation(c["원문"], c["번역"])
            return c, k, v, time.perf_counter() - t0

        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            futures = [pool.submit(_one, c, k) for c, k in todo]
            for fut in as_completed(futures):
                try:
                    c, k, v, secs = fut.result()
                except Exception as e:
                    print(f"실패(다시 실행하면 이 사례부터 이어서 함): {e}")
                    continue
                with lock:
                    _save(out, c, k, v, secs, runs)

    first = {cid: r for (cid, run), r in runs.items() if run == 1}
    if not first:
        print("저장된 결과가 없습니다.")
        return
    if "기대판정" in cases[0]:
        report_heldout(cases, first)
    else:
        report_human(cases, first)
    report_stability(runs)
    report_time(runs)


def _save(out, c, k, v, secs, runs):
    rec = {"id": c["번호"], "run": k, "human_severity": c.get("심각도", ""), "human_error": c.get("문제유형", ""),
           "expected": c.get("기대판정", ""), "injected": c.get("주입오류", ""),
           "verdict": v["verdict"], "meaning_score": v["meaning_score"], "issues": v["issues"],
           "source_claims": v["source_claims"], "additions": v["additions"],
           "semantic_similarity": v["semantic_similarity"], "judge_model": v["judge_model"],
           "seconds": round(secs, 2)}
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    runs[(c["번호"], k)] = rec
    exp = f"기대={c['기대판정']:6}" if c.get("기대판정") else f"사람={c.get('심각도') or '정상':2}"
    print(f"{c['번호']:>7} #{k} {exp} → {v['verdict']:6} {v['meaning_score']:5.0f} ({secs:.1f}초)")


if __name__ == "__main__":
    main()
