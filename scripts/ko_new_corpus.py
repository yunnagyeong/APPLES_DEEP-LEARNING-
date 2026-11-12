"""새 인물 코퍼스를 한→한 실험에 쓸 수 있게 준비한다 (D-097 ③, D-095 ②). 코드만, API 호출 없음.

입력: data/raw/corpus_<id>_raw.jsonl  (한 줄 = {id, title, type, text}, 사람이 정리한 원본. type은 essay·report 등)
하는 일(사람 id마다):
  1. (선택) --n-docs N: 글을 N편만 무작위로 뽑아 새 id "<id>_n<N>"으로 저장 → 우리 규모(12·18·28편)로 줄여 '양' 효과를 보는 용도
  2. 문단 경계로 예시 조각 만들기 (scripts/convert_data_to_examples.py와 같은 규칙) → data/examples/corpus_<id>_examples.jsonl
  3. 평가용 글 --heldout 편(기본 3, essay·report 중 무작위·시드 고정)을 빼고 학습용 data/examples/corpus_<id>_train.jsonl
  4. 평가 분할 기록: data/eval/split_info_new.json (기존 data/eval/split_info.json은 건드리지 않음)
쓰는 법(Mac, 프로젝트 루트):
  python scripts/ko_new_corpus.py prepare A B C                    # 전체 편수로
  python scripts/ko_new_corpus.py prepare A B C --n-docs 12        # 각 인물 12편으로 줄인 A_n12 B_n12 C_n12를 만든다
  LINGUA_CORPUS_SPLIT=train python scripts/build_index.py A_n12    # 임베딩(API 호출 있음, 인물마다)
  # 실험: 같은 환경변수 3개를 앞에 붙여 기존 스크립트를 그대로 쓴다
  LINGUA_PEOPLE=A_n12,B_n12,C_n12 LINGUA_SPLIT_INFO=data/eval/split_info_new.json LINGUA_CORPUS_SPLIT=train \\
    python scripts/ko_doc_transfer.py gen --set select --variant new12 && (같은 환경변수로) report --variant new12
주의: 새 인물은 문체 지시(P)가 없으므로 R만 쓴다. 현역 작가·블로거 글은 내부 분석용(커밋 금지, D-095).
      data/raw·data/examples의 새 파일은 .gitignore에 추가하거나 커밋하지 말 것.
"""
import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config
import convert_data_to_examples as cv

SEED = 42


def read_raw(pid):
    p = config.raw_path(pid)
    if not p.exists(): sys.exit(f"[중단] {p} 가 없습니다. (한 줄 = {{id,title,type,text}})")
    docs = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    bad = [d.get("id", "?") for d in docs if not all(k in d for k in ("id", "title", "text"))]
    if bad: sys.exit(f"[중단] {pid}: id/title/text가 없는 글 {bad[:5]}")
    return docs


def chunk(docs):
    rows = []
    for d in docs:
        t = d.get("type") or cv.infer_type(d["title"])
        for i, c in enumerate(cv.merge_pieces(cv.atomic_pieces(d["text"])), 1):
            rows.append({"id": f"{d['id']}_{i:02d}", "type": t, "topic": d["title"], "text": c})
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare(a):
    info_path = ROOT / "data/eval/split_info_new.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {"seed": SEED, "users": []}
    for pid in a.people:
        docs = read_raw(pid); out_id = pid
        if a.n_docs:
            if a.n_docs > len(docs): sys.exit(f"[중단] {pid}: 글 {len(docs)}편뿐인데 --n-docs {a.n_docs}")
            docs = sorted(random.Random(f"{SEED}-{pid}-{a.n_docs}").sample(docs, a.n_docs), key=lambda d: d["id"])
            out_id = f"{pid}_n{a.n_docs}"
            write_jsonl(config.raw_path(out_id), docs)
        rows = chunk(docs)
        write_jsonl(config.EXAMPLES_DIR / f"corpus_{out_id}_examples.jsonl", rows)
        pool = [d["id"] for d in docs if (d.get("type") or cv.infer_type(d["title"])) in ("essay", "report")]
        if len(pool) <= a.heldout + 2: sys.exit(f"[중단] {out_id}: essay·report가 {len(pool)}편뿐이라 평가용 {a.heldout}편을 빼면 학습이 너무 적습니다.")
        held = sorted(random.Random(f"{SEED}-{out_id}-held").sample(pool, a.heldout))
        train = [r for r in rows if r["id"].rsplit("_", 1)[0] not in held]
        write_jsonl(config.EXAMPLES_DIR / f"corpus_{out_id}_train.jsonl", train)
        title = {d["id"]: d["title"] for d in docs}
        info["users"] = [u for u in info["users"] if u["user"] != out_id] + [{
            "user": out_id, "total_docs": len(docs), "train_docs": len(docs) - len(held), "train_chunks": len(train),
            "heldout_docs": {h: title[h] for h in held}}]
        chars = sum(len(d["text"]) for d in docs)
        print(f"{out_id}: 글 {len(docs)}편({chars:,}자) / 예시 {len(rows)}개 / 평가용 {held} / 학습 예시 {len(train)}개")
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {info_path.relative_to(ROOT)}\n다음: 인물마다 LINGUA_CORPUS_SPLIT=train python scripts/build_index.py <id>  (임베딩 API 호출)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare"); p.add_argument("people", nargs="+"); p.add_argument("--n-docs", type=int, default=0)
    p.add_argument("--heldout", type=int, default=3)
    a = ap.parse_args(); {"prepare": prepare}[a.cmd](a)
