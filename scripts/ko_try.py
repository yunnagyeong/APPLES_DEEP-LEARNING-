"""원문 한 편을 넣고 → 목표 인물 문체로 변환 → 분류기 두 개(SVM·Delta)로 판정.
사용(Mac, .venv):  python scripts/ko_try.py --file my.txt --target suyoung --mode P+R
                   python scripts/ko_try.py --text "문단..." --target ganghee --mode R
--mode N 은 변환 없이 입력문 그대로 판정. 실제 API를 호출하므로 Mac에서만 실행.
분류기는 150~450자 문단 기준으로 학습됐다. 입력이 문단보다 훨씬 길면 --split 으로 빈 줄 단위로 나눠 각각 판정한다."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter
from src import ko_style, ko_classify as kc

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--text"); g.add_argument("--file")
ap.add_argument("--target", required=True, choices=list(ko_style.PEOPLE))
ap.add_argument("--mode", default="P+R", choices=["N", "P", "R", "P+R"])
ap.add_argument("--version", default="new", choices=list(ko_style.PROMPT_VERSIONS))
ap.add_argument("--split", action="store_true", help="빈 줄 단위 문단으로 나눠 각각 변환·판정")
a = ap.parse_args()
text = a.text if a.text else Path(a.file).read_text(encoding="utf-8")
paras = [p.strip() for p in text.split("\n\n") if p.strip()] if a.split else [text.strip()]

clf = kc.Classifiers()
outs = []
for p in paras:
    if a.mode == "N": outs.append(p); continue
    r = ko_style.convert_style(p, a.target, a.mode, version=a.version)
    outs.append(r["converted"])
    print(f"[변환 {a.mode}·{a.version}→{a.target}] 지시 {len(r['lines'])}줄\n{r['converted']}\n")
try:
    from src import ko_format
    for p, o in zip(paras, outs):
        c = ko_format.check(p, o)
        print("형식 검사:", "통과" if c["pass"] else "표시 · " + "; ".join(c["reasons"]))
except Exception as e:
    print("형식 검사 생략:", e)
for name, texts in (("입력(원문)", paras), ("출력", outs)):
    pr = clf.predict(texts)
    print(f"{name:<8} SVM {dict(Counter(pr['SVM']))}  Delta {dict(Counter(pr['Delta']))}   (목표 {a.target}, 문단 {len(texts)}개)")
