"""평가 입력 만들기 (D-085): 코퍼스 문서 제목을 주제로 AI가 해라체 글 한 편(2,000자 이상)을 새로 쓴다.

사람이 주제를 고르지 않는다. 세 사람의 essay·report 제목 중 내용이 없는 제목(예: "4주차 보강과제")과
살아 있는 실존 인물 이름이 든 제목은 뺀 뒤, 시드로 섞어 선택용 20편·확인용 40편으로 나눈다.
  → data/ko_docs/select/*.txt, data/ko_docs/confirm/*.txt, data/ko_docs/meta.json(입력 → 주제 출처 문서)
출처 문서는 ko_doc_transfer.py가 그 사람을 목표로 바꿀 때 예시 검색에서 뺀다(같은 내용의 원문을 베끼는 것 방지).

  python scripts/ko_make_inputs.py --set select [--dry]     # 20편 (생성 20회)
  python scripts/ko_make_inputs.py --set confirm            # 40편 (모든 설정을 고정한 뒤에)
"""
import argparse, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "ko_docs"
from src.ko_style import PEOPLE  # LINGUA_PEOPLE로 바꿀 수 있음
SKIP = {("ganghee", "doc_006"),   # 4주차 보강과제(주제 없음)
        ("nakyung", "doc_001"),   # 살아 있는 실존 인물(배우) 이름
        ("suyoung", "doc_008")}   # 살아 있는 실존 인물(정치인) 이름
N_SELECT, N_CONFIRM, SEED, MIN_CHARS = 20, 40, "inputs", 2000


def topics():
    from src import config
    out = []
    for u in PEOPLE:
        for l in config.RAW_DIR.joinpath(f"corpus_{u}_raw.jsonl").open(encoding="utf-8"):
            if not l.strip(): continue
            o = json.loads(l)
            if o.get("type") in ("essay", "report") and (u, o["id"]) not in SKIP:
                out.append({"person": u, "doc": o["id"], "type": o["type"], "title": str(o.get("title", "")).strip()})
    out.sort(key=lambda x: (x["person"], x["doc"])); random.Random(SEED).shuffle(out)
    return {"select": out[:N_SELECT], "confirm": out[N_SELECT:N_SELECT + N_CONFIRM]}


def prompt(t):
    kind = "에세이" if t["type"] == "essay" else "보고서"
    return (f"다음 주제로 대학생 과제 {kind}를 한국어로 써 주세요.\n"
            f"- 분량: 2,200~2,800자, 문단 5~8개, 문단 사이는 빈 줄\n"
            f"- 문체: '-다'로 끝나는 문어체(합니다체 쓰지 않음), 줄글(제목·소제목·목록·표 없이 본문만)\n"
            f"- 실제 자료를 인용하는 척하거나 출처·숫자를 지어내지 말 것\n\n주제: {t['title']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["select", "confirm"], required=True); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    from src import config, ko_style
    plan = topics()[a.set]
    meta_p = OUT / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    d = OUT / a.set; d.mkdir(parents=True, exist_ok=True)
    print(f"[{a.set}] {len(plan)}편 / 모델 {config.GENERATION_MODEL}")
    if a.dry:
        for t in plan: print(f"  {t['person']} {t['doc']} {t['type']} | {t['title']}")
        print("\n" + prompt(plan[0])); return
    for t in plan:
        name = f"{t['person']}_{t['doc']}"
        f = d / f"{name}.txt"
        if f.exists(): continue
        try:
            text = ko_style.generate(prompt(t), "입력 글 생성")
        except Exception as e:
            print(f"{name} ERR {type(e).__name__}: {str(e)[:100]}"); continue
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        f.write_text(text + "\n", encoding="utf-8")
        meta[name] = {**t, "set": a.set, "model": config.GENERATION_MODEL, "chars": len(text)}
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name} {len(text)}자" + ("  ⚠2,000자 미만" if len(text) < MIN_CHARS else ""))


if __name__ == "__main__":
    main()
