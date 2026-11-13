"""한→한 변환 → 검증 → FAIL일 때만 재번역 (src/ko_pipeline.py) 테스트. API 호출 없음(생성 모델을 가짜로 바꿈).

    python test_ko_pipeline.py        # 또는  python -m pytest test_ko_pipeline.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import ko_pipeline, ko_style, ko_verify  # noqa: E402

SRC = ("원격 근무는 지난 3년 사이 크게 늘었다. 많은 기업이 이 방식을 받아들였지만, 모든 기업이 그런 것은 아니다. "
       "직원들은 출퇴근 시간이 줄어든 점을 장점으로 꼽는다. 다만 협업이 어려워질 수 있다는 우려도 있다.")
OK = ("원격 근무는 최근 3년 동안 부쩍 늘었다. 많은 기업이 이 방식을 받아들였으나, 모든 기업이 그런 건 아니다. "
      "직원들은 출퇴근 시간이 줄었다는 점을 장점으로 든다. 그렇지만 협업이 어려워질 수 있다는 걱정도 있다.")
EXAMPLES = [{"id": "doc_003_p01", "score": 0.8, "type": "짝",
             "ai": "온라인 수업은 학생들에게 많은 편의를 제공한다고 할 수 있다.",
             "text": "온라인 수업은 학생들에게 편하다. 집에서 강의를 듣고 과제를 내는 데 아무 문제가 없었다고 나는 생각한다."}]
COPIED = "집에서 강의를 듣고 과제를 내는 데 아무 문제가 없었다고 나는 생각한다."


class FakeGen:
    """ko_style.generate 대신 쓰는 가짜. 정해 둔 출력을 차례로 돌려주고 받은 프롬프트를 기록한다."""

    def __init__(self, *outputs, fail_on=None):
        self.outputs, self.prompts, self.fail_on = list(outputs), [], fail_on

    def __call__(self, prompt, label="문체 변환"):
        self.prompts.append(prompt)
        if self.fail_on is not None and len(self.prompts) == self.fail_on:
            raise RuntimeError("503 UNAVAILABLE (가짜)")
        return self.outputs.pop(0)


def run(*outputs, fail_on=None, **kw):
    fake = FakeGen(*outputs, fail_on=fail_on)
    orig = ko_style.generate
    ko_style.generate = fake
    try:
        r = ko_pipeline.convert_checked(SRC, "ganghee", "R", examples=EXAMPLES, **kw)
    finally:
        ko_style.generate = orig
    return r, fake


def test_pass_is_left_alone():
    r, fake = run(OK)
    c = r["check"]
    assert c["status"] == "PASS" and c["first_verdict"] == "PASS"
    assert len(fake.prompts) == 1 and c["retries"] == 0 and len(c["attempts"]) == 1
    assert r["converted"] == OK and c["review"] == []


def test_review_is_not_retried_but_marked():
    out = OK.replace("받아들였으나", "받아들이지 않았으나")  # 부정 뒤집힘(major)
    r, fake = run(out)
    c = r["check"]
    assert c["status"] == "REVIEW" and len(fake.prompts) == 1 and c["retries"] == 0
    assert r["converted"] == out  # 고치지 않음
    assert any("부정 뒤집힘" in p["labels"] for p in c["review"])
    p = next(p for p in c["review"] if "부정 뒤집힘" in p["labels"])
    assert p["where"] == "output" and p["sentence"] in out  # 변환문에서 강조할 수 있는 문장
    segs = ko_verify.mark_segments(out, [p["sentence"]])
    assert "".join(s for s, _ in segs) == out and any(m for _, m in segs)


def test_fail_number_is_retranslated_with_feedback():
    bad = OK.replace("3년", "5년")
    r, fake = run(bad, OK)
    c = r["check"]
    assert c["first_verdict"] == "FAIL" and c["status"] == "PASS"
    assert c["retries"] == 1 and c["best_attempt"] == 2 and r["converted"] == OK
    fix = fake.prompts[1]
    assert "숫자 불일치" in fix and "원문 숫자 3 없음" in fix and bad in fix  # 무엇을 고칠지 알려 줌
    assert EXAMPLES[0]["text"] not in fix  # 재번역 프롬프트엔 예시를 넣지 않음
    assert c["first_output"] == bad


def test_fail_meta_is_fixed_by_rule_without_api():
    r, fake = run("다음은 강희 문체로 다시 쓴 글입니다.\n\n" + OK)
    c = r["check"]
    assert c["first_verdict"] == "FAIL" and c["status"] == "PASS"
    assert len(fake.prompts) == 1 and c["retries"] == 0  # 추가 API 호출 없음
    assert c["attempts"][1]["how"] == "머리말 규칙 삭제" and not c["attempts"][1]["api_call"]
    assert r["converted"] == OK


def test_fail_example_copy_is_retranslated():
    bad = OK + " " + COPIED
    r, fake = run(bad, OK)
    c = r["check"]
    assert c["first_verdict"] == "FAIL" and "검색 예시 베낌" in c["attempts"][0]["summary"]
    assert c["status"] == "PASS" and c["retries"] == 1
    fix = fake.prompts[1]
    assert "검색 예시 베낌" in fix and "아무 문제가 없었다고" in fix  # 베낀 구절을 짚어 줌


def test_empty_output_regenerates_same_prompt():
    r, fake = run("", OK)
    assert r["check"]["first_verdict"] == "FAIL" and r["check"]["status"] == "PASS"
    assert fake.prompts[1] == fake.prompts[0]  # 같은 프롬프트


def test_still_fail_after_max_retries_keeps_best_and_warns():
    bad1, bad2 = OK.replace("3년", "5년"), OK.replace("3년", "7년")
    r, fake = run(bad1, bad2)
    c = r["check"]
    assert c["status"] == "FAIL" and c["retries"] == 1 and len(fake.prompts) == 2
    assert c["best_attempt"] == 1 and r["converted"] == bad1  # 둘 다 FAIL이면 앞 시도
    assert "반드시 확인" in c["message"]


def test_retry_can_end_in_review():
    bad = OK.replace("3년", "5년")
    half = OK.replace("받아들였으나", "받아들이지 않았으나")
    r, _ = run(bad, half)
    c = r["check"]
    assert c["status"] == "REVIEW" and c["best_attempt"] == 2 and c["review"]


def test_max_retries_zero_means_no_retry():
    r, fake = run(OK.replace("3년", "5년"), max_retries=0)
    assert r["check"]["status"] == "FAIL" and len(fake.prompts) == 1


def test_two_retries_allowed():
    r, fake = run(OK.replace("3년", "5년"), OK.replace("3년", "6년"), OK, max_retries=2)
    c = r["check"]
    assert c["status"] == "PASS" and c["retries"] == 2 and c["best_attempt"] == 3 and len(fake.prompts) == 3


def test_retry_error_returns_first_result():
    r, _ = run(OK.replace("3년", "5년"), fail_on=2)
    c = r["check"]
    assert c["status"] == "FAIL" and "재번역 호출 실패" in c["note"] and len(c["attempts"]) == 1


def test_dummy_mode_never_calls_api():
    fake = FakeGen()
    orig, ko_style.generate = ko_style.generate, fake
    try:
        r = ko_pipeline.convert_checked(SRC, "ganghee", "R", dummy_output=OK.replace("3년", "5년"))
        r2 = ko_pipeline.convert_checked(SRC, "ganghee", "R", dummy_output="다시 쓴 글: " + OK)
    finally:
        ko_style.generate = orig
    assert fake.prompts == []
    assert r["check"]["status"] == "FAIL" and "가짜 결과" in r["check"]["note"]
    assert r2["check"]["status"] == "PASS"  # 규칙 삭제는 가짜 모드에서도 됨


def test_history_line():
    r, _ = run(OK.replace("3년", "5년"), OK)
    assert ko_pipeline.history_line(r["check"]) == "FAIL → 지적 부분만 고쳐 다시 생성 → PASS (최종: 2번째 시도)"


def test_strip_meta_keeps_real_sentences():
    for legit in ["정책을 바꾼 이유는 분명하다. " + SRC, "결과: 실험은 성공했다."]:
        assert ko_verify.strip_meta(legit, legit) == legit.strip()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    sys.exit(1 if failed else 0)
