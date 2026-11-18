"""예시 베낌 FAIL 건에 기존 재번역(ko_pipeline 최소 수정 프롬프트, 예시 없음)을 한 번 적용해 결과를 비교한다 (11/19, B 세션).
생성 API 호출 = 건수(기본 9). 결과: results/ko_doc/retry_check.jsonl (이어서 실행 가능).
    .venv/bin/python scripts/ko_retry_check.py            # 실행
    .venv/bin/python scripts/ko_retry_check.py --dry      # 대상만 출력
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import ko_verify, ko_pipeline, ko_style

CASES = [("new18_para", "leesang_n18_doc_008", "leegwangsu_n18"), ("new18_para", "leesang_n18_doc_015", "leesang_n18"),
         ("new18_para", "leesang_n18_doc_015", "leeyuksa_n18"), ("new18_para", "leesang_n18_doc_016", "leeyuksa_n18"),
         ("new18_para", "leeyuksa_n18_doc_017", "leesang_n18"), ("team_para", "leegwangsu_n18_doc_003", "nakyung"),
         ("team_para", "leegwangsu_n18_doc_003", "suyoung"), ("team_para", "leesang_n18_doc_010", "nakyung"),
         ("team_para", "leeyuksa_n18_doc_017", "ganghee")]
OUT = ROOT / "results/ko_doc/retry_check.jsonl"

def main(dry=False):
    done = {(r["variant"], r["doc"], r["target"]) for r in map(json.loads, OUT.open(encoding="utf-8"))} if OUT.exists() else set()
    for v, d, t in CASES:
        if (v, d, t) in done: continue
        gen = {(r["doc"], r["target"]): r for r in map(json.loads, (ROOT / f"results/ko_doc/{v}/gen.jsonl").open(encoding="utf-8"))}[(d, t)]
        ret = json.loads((ROOT / f"results/ko_doc/{v}/retrieved.json").read_text(encoding="utf-8"))[f"select|{d}|{t}|{gen['k']}|0"]
        src = (ROOT / f"data/ko_docs/select/{d}.txt").read_text(encoding="utf-8")
        before = ko_verify.verify(src, gen["output"], ret)
        print(f"[{v}] {d} → {t}: 전 {before['verdict']} | {ko_verify.summarize(before)[:120]}")
        if dry: continue
        prompt = ko_pipeline.fix_prompt(src, gen["output"], before)
        new = ko_style.generate(prompt, "재번역 점검")
        after = ko_verify.verify(src, new, ret)
        print(f"    후 {after['verdict']} | {ko_verify.summarize(after)[:120]}")
        rec = {"variant": v, "doc": d, "target": t, "before_verdict": before["verdict"], "after_verdict": after["verdict"],
               "before_summary": ko_verify.summarize(before), "after_summary": ko_verify.summarize(after),
               "copy_evidence": [i.get("evidence") or i["note"] for i in before["issues"] if i["type"] == "EXAMPLE_COPY"],
               "before": gen["output"], "after": new, "prompt": prompt}
        with OUT.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main("--dry" in sys.argv)
