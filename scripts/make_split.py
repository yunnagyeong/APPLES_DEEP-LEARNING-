"""평가용/학습용 코퍼스 분리 (A/B/C 비교 실험 준비).

사용법 (프로젝트 루트): python scripts/make_split.py
- 사람마다 글(doc) 3편 이상을 '통째로' 평가용으로 빼고, 그 글들에서 문단 10개를 뽑습니다.
  (같은 글의 문단이 평가용과 학습용에 나뉘어 들어가면 결과를 믿을 수 없으므로 글 단위로 분리)
- 한 글에서 최대 4문단, 200자 미만 문단은 제외 → 한 주제로 쏠리지 않게 함.
- 시드 고정이라 누가 돌려도 같은 결과가 나옵니다.

만드는 파일:
  data/eval/eval_sources.jsonl            평가 원문 30개 (사람당 10개)
  data/eval/split_info.json               어떤 글을 뺐는지 기록 (발표·포스터용)
  data/examples/corpus_<user>_train.jsonl 평가용 글을 뺀 학습용 코퍼스
  data/examples/corpus_<user>_train_index_cache.json
      기존 임베딩 캐시에서 학습용 문단만 복사 → 임베딩 API를 다시 부르지 않음
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "data" / "examples"
EVAL = ROOT / "data" / "eval"

USERS = ["ganghee", "nakyung", "suyoung"]
SEED, N, MINLEN, MIN_DOCS, PER_DOC = 42, 10, 200, 3, 4


def doc_of(chunk_id: str) -> str:
    return chunk_id.rsplit("_", 1)[0]  # doc_019_18 → doc_019


def main():
    rng = random.Random(SEED)
    EVAL.mkdir(parents=True, exist_ok=True)
    eval_rows, report = [], []

    for user in USERS:
        rows = [json.loads(l) for l in (EX / f"corpus_{user}_examples.jsonl").open(encoding="utf-8") if l.strip()]
        docs = defaultdict(list)
        for r in rows:
            docs[doc_of(r["id"])].append(r)
        eligible = {d: [c for c in cs if len(c["text"]) >= MINLEN] for d, cs in docs.items()}

        order = sorted(docs)
        rng.shuffle(order)
        held = []
        for d in order:
            ok = sum(min(PER_DOC, len(eligible[h])) for h in held)
            if len(held) >= MIN_DOCS and ok >= N:
                break
            held.append(d)

        per = {h: rng.sample(eligible[h], min(PER_DOC, len(eligible[h]))) for h in held}
        picked, i = [], 0
        while len(picked) < N:  # 글끼리 돌아가며 뽑기
            for h in held:
                if i < len(per[h]) and len(picked) < N:
                    picked.append(per[h][i])
            i += 1
        picked.sort(key=lambda c: c["id"])
        for c in picked:
            eval_rows.append({"id": f"{user}_{c['id']}", "person": user, "doc": doc_of(c["id"]),
                              "type": c["type"], "topic": c["topic"], "text": c["text"]})

        train = [r for r in rows if doc_of(r["id"]) not in held]
        with (EX / f"corpus_{user}_train.jsonl").open("w", encoding="utf-8") as f:
            for r in train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 임베딩 캐시 재사용
        full_cache_path = EX / f"corpus_{user}_index_cache.json"
        if full_cache_path.exists():
            full = json.loads(full_cache_path.read_text(encoding="utf-8"))
            sub = {r["id"]: full[r["id"]] for r in train if r["id"] in full}
            (EX / f"corpus_{user}_train_index_cache.json").write_text(
                json.dumps(sub, ensure_ascii=False), encoding="utf-8")
            reused = len(sub)
        else:
            reused = 0

        assert not ({doc_of(r["id"]) for r in train} & set(held)), "평가용 글이 학습용에 섞였습니다"
        report.append({"user": user, "total_docs": len(docs), "train_docs": len(docs) - len(held),
                       "train_chunks": len(train), "embeddings_reused": reused,
                       "heldout_docs": {h: docs[h][0]["topic"] for h in sorted(held)},
                       "eval_chunks_per_doc": {h: sum(1 for c in picked if doc_of(c["id"]) == h) for h in held}})

    with (EVAL / "eval_sources.jsonl").open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (EVAL / "split_info.json").write_text(json.dumps(
        {"seed": SEED, "per_user": N, "min_chars": MINLEN, "min_docs": MIN_DOCS, "max_per_doc": PER_DOC,
         "users": report}, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in report:
        print(f"{r['user']}: 평가용 글 {len(r['heldout_docs'])}편 / 학습용 {r['train_chunks']}문단 "
              f"(임베딩 {r['embeddings_reused']}개 재사용)")
    print(f"→ data/eval/eval_sources.jsonl {len(eval_rows)}개")


if __name__ == "__main__":
    main()
