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


def test_old_spelling_negation_is_negation():
    """'아니하다'(옛 표기, 이광수 등)도 부정으로 본다. 뜻이 같으면 PASS, 부정이 사라지면 REVIEW."""
    src = "이 방법은 효과가 크지 않다. 많은 사람이 이를 받아들이지 않았다. 그래도 연구는 계속되었다."
    same = "이 방법은 효과가 크지 아니하다. 많은 사람이 이를 받아들이지 아니하였다. 그래도 연구는 계속되었다."
    lost = "이 방법은 효과가 크다. 많은 사람이 이를 받아들이지 아니하였다. 그래도 연구는 계속되었다."
    assert ko_verify.verify(src, same)["verdict"] == "PASS"
    assert any(i["type"] == "NEGATION_FLIP" for i in ko_verify.verify(src, lost)["issues"])


def _types(a, b):
    return {i["type"] for i in ko_verify.verify(a, b, check_format=False)["issues"] if i["severity"] != "minor"}


# 문체를 바꿔도 뜻이 같은 쌍(오탐이면 안 됨) — 파트너 검토지(new18h_para)에서 나온 실제 유형
SAME_MEANING = [
    ("이는 중요한 과정이다.", "이는 중요한 과정이 아니겠는가."),                                   # 부정 수사 의문 = 긍정
    ("인간은 한계 속에 갇혀 살아가지만, 영화는 경험을 제공한다.",
     "인간은 한계 속에 갇혀 살아가지 않으랴마는, 영화는 경험을 제공한다."),                       # 문장 가운데 수사 의문
    ("이 문제는 꼭 해결해야 한다.", "이 문제는 해결하지 않을 수 없다."),                         # 이중부정 = 강한 긍정
    ("이는 차이를 넘어 시선이다.", "이는 단순한 차이에 그치지 않고 시선이다."),                       # 대조 '그치지 않고'
    ("이는 문제를 넘어 구조의 문제다.", "이는 단순한 문제가 아니라 구조의 문제다."),                   # 대조 'A가 아니라 B'
    ("그는 결국 무너진다.", "그는 결국 무너지고 만다."),                                             # '-고 말다' ≠ 금지
    ("그것은 한낱 핑계일 뿐이다.", "그것은 한낱 핑계에 지나지 않는다."),                               # '지나지 않다' = 뿐이다
    ("상해는 거대한 현장과 같다.", "상해는 거대한 현장과 다를 바가 없다."),                             # 굳은 말
    ("문화를 꽃피울 수 없다.", "문화를 꽃피울 수가 없다."),                                           # 조사만 다름
    ("그는 맥락을 고려하지 못하고 하나에 집중한다.", "그는 맥락을 굽어볼 줄 모르고 하나에 집중한다."),   # '줄 모르다' = 못하다
    ("감정은 순종, 경외, 혹은 무감각인 경우가 많다.", "감정은 순종이요 경외요, 아니면 무감각인 경우가 허다하다."),  # 아니면 = 또는
    ("서사는 무력함을 그대로 인정한다.", "서사는 무력함을 가감 없이 그대로 인정한다."),                       # 낱말 속 '없'
    ("사랑은 인격적 과업이라 할 수 있다.", "사랑은 인격적 과업이라 하겠다."),                           # 결론 말투
    ("예술성을 뒷받침할 수 있는 제작 환경을 조성해야 한다.", "예술성을 북돋워 줄 제작 환경을 마련해야 한다."),  # 능력 '수 있는'
    ("이야기는 단면을 그대로 드러낸다.", "이야기는 단면을 그대로 드러내 보이는 것이다."),               # 보조 '보이다'
    ("조언을 들었다면 좋았겠다.", "조언을 들을 수 있었다면 좋았겠다."),                                 # 과거 능력·가정
    ("그는 언제나 성실하다.", "그는 늘 성실하다."),                                                     # 같은 뜻 범위어
]

# 진짜 뜻 변화(놓치면 안 됨)
MEANING_CHANGED = [
    ("이 방법은 효과가 있다.", "이 방법은 효과가 없다.", "NEGATION_FLIP"),
    ("그는 결과를 받아들였다.", "그는 결과를 받아들이지 않았다.", "NEGATION_FLIP"),
    ("그러나 상처받지 않기 위해 성벽을 쌓을수록 교류는 차단된다.",
     "그러나 상처받기 위해 성벽을 쌓을수록 교류는 차단된다.", "NEGATION_FLIP"),
    ("아이는 풍경이 다름을 이해하지 못하고 그저 자기 모습을 고른다.",
     "아이는 풍경이 다름을 이해하고 그저 자기 모습을 고른다.", "NEGATION_FLIP"),                    # '그저'가 있어도 '못'은 셈
    ("그러나 상처받지 않기 위해 성벽을 쌓을수록 교류는 차단된다.",
     "그러나 상처받지 않기 위해 성벽을 쌓을수록 교류는 차단되는 것을 어찌하리오.", None),       # 끝 수사 의문은 앞 부정을 지우지 않음(뒤집힘 없음)
    ("이 정책은 효과를 낸다.", "이 정책은 효과를 낼 수 있다.", "MODALITY_SHIFT"),
    ("이 정책은 효과를 낸다.", "이 정책은 효과를 내는 모양이다.", "MODALITY_SHIFT"),                   # '모양이다' = 추측
    ("결과가 좋다.", "결과가 좋아 보인다.", "MODALITY_SHIFT"),                                          # 형용사+어 보이다 = 추측
    ("사람들은 그를 신뢰한다.", "모든 사람들은 그를 신뢰한다.", "SCOPE_CHANGE"),
]


def test_style_changes_with_same_meaning_pass():
    bad = [(a, b, _types(a, b)) for a, b in SAME_MEANING if _types(a, b)]
    assert not bad, bad


def test_real_meaning_changes_are_caught():
    missed = [(a, b, want) for a, b, want in MEANING_CHANGED if want and want not in _types(a, b)]
    wrong = [(a, b) for a, b, want in MEANING_CHANGED if want is None and "NEGATION_FLIP" in _types(a, b)]
    assert not wrong, wrong
    assert not missed, missed


def test_merged_sentences_keep_negation_check():
    """두 문장을 한 문장으로 합쳐도 부정 비교는 합친 두 문장 전체와 한다(합친 쪽에서 부정이 생기면 잡음)."""
    src = "근무 환경이 피로의 원인이 될 수 있다. 따라서 조직은 응답을 요구하지 않는 규범을 마련해야 한다."
    same = "근무 환경이 피로의 원인이 될 수 있으므로, 따라서 조직은 응답을 요구하지 않는 규범을 마련해야 한다."
    flip = "근무 환경이 피로의 원인이 될 수 있으므로, 따라서 조직은 응답을 요구하지 않는 규범을 마련해야 하지 않는다."
    assert "NEGATION_FLIP" not in _types(src, same)
    assert "NEGATION_FLIP" in _types(src, flip)


def test_native_vocabulary_rewrite_is_not_example_copy():
    """고유어로 많이 바꾸는 문체(경험하다→겪다)는 새 낱말이 예시에 있어도 원문 문장을 바꿔 쓴 것이면 베낌이 아니다.
    원문과 상관없는 예시 문장을 끼워 넣으면 베낌(FAIL)."""
    src = "인간은 태어나서 죽음에 이르기까지 다양한 형태로 사랑을 경험한다. 그 과정에서 삶의 의미를 재구성한다."
    out = "인간은 나면서 죽음에 이르기까지 여러 가지 모양으로 사랑을 겪는다. 그 길에서 삶의 뜻을 다시 짓는다."
    ex = [{"id": "x", "text": "나는 여러 가지 모양으로 세상을 겪었다. 그 길에서 뜻을 다시 지었다. 바닷가 마을의 겨울 바람은 몹시 매서웠다."}]
    assert "EXAMPLE_COPY" not in {i["type"] for i in ko_verify.verify(src, out, ex, check_format=False)["issues"]}
    inserted = out + " 바닷가 마을의 겨울 바람은 몹시 매서웠다."
    assert ko_verify.verify(src, inserted, ex, check_format=False)["verdict"] == "FAIL"


def test_sino_numerals_in_hangul_match_digits():
    """'100년'→'백 년', '2호선'→'이 호선'은 같은 수. 다른 수로 바뀌면 FAIL."""
    assert ko_verify.check_numbers("100년이 넘었다. 2호선을 탔다.", "백 년이 넘었다. 이 호선을 탔다.") == []
    assert any(i["type"] == "NUMBER_MISMATCH" for i in ko_verify.check_numbers("100년이 넘었다.", "이백 년이 넘었다."))


def test_truncated_retry_is_not_chosen():
    """재번역이 마지막 문단만 돌려주면(잘림) 그 시도는 FAIL이라 고르지 않고 앞 글을 남긴다 (11-19 파트너 사례)."""
    bad = OK + " " + COPIED
    cut = "그렇지만 협업이 어려워질 수 있다는 걱정도 있다."
    r, fake = run(bad, cut)
    c = r["check"]
    summ = c["attempts"][1]["summary"]
    assert c["attempts"][1]["verdict"] == "FAIL" and ("글이 잘림" in summ or "전체 내용 이탈" in summ)
    assert c["best_attempt"] == 1 and r["converted"] == bad


def test_length_ratio_marks_truncation():
    src = " ".join([SRC] * 3)  # 200자 이상일 때만 길이 비를 본다
    assert "TRUNCATED" in {i["type"] for i in ko_verify.verify(src, OK)["issues"]}
    assert "TRUNCATED" not in {i["type"] for i in ko_verify.verify(src, " ".join([OK] * 3))["issues"]}


def test_truncated_first_output_regenerates_same_prompt():
    cut = "원격 근무는 최근 3년 동안 부쩍 늘었다."
    r, fake = run(cut, OK)
    assert r["check"]["first_verdict"] == "FAIL" and r["check"]["status"] == "PASS"
    assert fake.prompts[1] == fake.prompts[0]  # 잘린 글은 고칠 곳 지시가 아니라 처음부터 다시


def test_example_entity_misparse_is_review_not_fail():
    """Kiwi가 옛 어미·한자 읽기를 고유명사로 잘못 붙인 것('이로소이다'의 '로소', '別離[별리]')은 베낌 FAIL이 아님.
    예시에 있던 세 글자 이상 이름을 어절 첫머리에 그대로 쓰면 베낌(FAIL)."""
    src = "우리 생활에는 여러 가지 문제가 있다. 이 문제는 앞으로 더 깊이 살펴볼 만한 주제다."
    ex = [{"id": "x", "text": "이 일은 깊이 검토할 대상이로소이다. 그날의 別離[별리]는 슬펐다. 정지용은 바다를 노래했다."}]
    for out in ("우리 생활에는 여러 가지 문제가 있도다. 이 문제는 앞으로 더 깊이 살펴볼 대상이로소이다.",
                "우리 생활에는 여러 가지 別離[별리]의 문제가 있다. 이 문제는 앞으로 더 깊이 살펴볼 만한 주제다."):
        assert "EXAMPLE_COPY" not in {i["type"] for i in ko_verify.verify(src, out, ex, check_format=False)["issues"]}, out
    copied = "우리 생활에는 정지용이 말한 여러 가지 문제가 있다. 이 문제는 앞으로 더 깊이 살펴볼 만한 주제다."
    assert "EXAMPLE_COPY" in {i["type"] for i in ko_verify.verify(src, copied, ex, check_format=False)["issues"]}


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
