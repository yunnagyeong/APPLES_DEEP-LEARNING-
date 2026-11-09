import json
import re
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Lingua Me",
    layout="wide",
)

ROOT = Path(__file__).parent
EVAL_DIR = ROOT / "data" / "eval"   # 팀원 실제 영어 글: english_<user>.jsonl
PERSONAS = {"나경": "nakyung", "강희": "ganghee", "수영": "suyoung"}
CONDITIONS = {
    "A": "A · 일반 번역",
    "B": "B · 단순 개인화 (RAG)",
    "C": "C · 우리 팀 방법",
}


# ==================================================
# 번역기 연결 (src/ 파이프라인)
#   A: src.baselines.translate_zero_shot     (일반 번역)
#   B: src.baselines.translate_in_context    (RAG 예시만)
#   C: src.pipeline.translate_with_verification
#      (Style Profile + RAG + 보존 규칙 → 의미 검증 v2 → 문제 구절만 고치는 재번역, 가장 좋은 시도 반환)
# 사이드바에서 "데모용 가짜 결과"를 켜면 API 호출 없이 화면만 확인할 수 있습니다.
#
# 속도: A·B·C를 동시에 실행(전체 대기 = 가장 느린 C), 검색은 한 번만 해서 B·C가 공유,
#       Kiwi·스타일 프로필은 처음 한 번만 준비. C 재번역은 최대 config.DEFAULT_MAX_RETRIES회.
# ==================================================


@st.cache_resource(show_spinner=False)
def load_style_profile(user):
    from src.pipeline import _load_or_compute_style_profile
    return _load_or_compute_style_profile(user)


@st.cache_resource(show_spinner="형태소 분석기·스타일 프로필 준비 중... (처음 한 번만)")
def warm_up(user):
    from src import style_features as sf
    sf._get_kiwi().tokenize("준비")
    load_style_profile(user)
    return True


def translate(korean_text, user, condition, examples=None):
    """번역 결과 dict: {"text": 영어 번역, "info": 짧은 설명, "detail": C의 검증 결과(없으면 None)}"""
    if condition == "A":
        from src.baselines import translate_zero_shot
        return {"text": translate_zero_shot(korean_text), "info": "", "detail": None}
    if condition == "B":
        from src.baselines import translate_in_context
        return {"text": translate_in_context(korean_text, user, examples=examples), "info": "", "detail": None}
    from src.pipeline import translate_with_verification
    from src import config
    result = translate_with_verification(
        korean_text, load_style_profile(user), user, max_retries=config.DEFAULT_MAX_RETRIES, examples=examples,
    )
    info = f"검증 {result['verdict']} · Meaning {result['meaning_score']:.0f} · 시도 {result['attempts']}회"
    if result["attempts"] > 1:
        info += f"(채택: {result['best_attempt']}번째)"
    return {"text": result["translation"], "info": info, "detail": result}


def run_all_conditions(source, user, use_dummy):
    """A·B·C를 동시에 실행. (results, errors) 반환. 스레드 안에서는 st.* 를 부르지 않습니다."""
    from concurrent.futures import ThreadPoolExecutor

    def timed(c, examples):
        start = time.time()
        r = {"text": DUMMY_OUTPUTS[c], "info": "가짜 결과"} if use_dummy else translate(source, user, c, examples)
        r["info"] = (r["info"] + " · " if r["info"] else "") + f"{time.time() - start:.1f}초"
        return r

    results, errors = {}, {}
    examples = None
    if not use_dummy:
        try:
            from src import config
            from src.retrieve import retrieve_similar
            examples = retrieve_similar(source, user, k=config.DEFAULT_TOP_K)
        except BaseException:  # 실패하면 B·C가 각자 검색하도록 둠
            examples = None
    with ThreadPoolExecutor(max_workers=len(CONDITIONS)) as pool:
        futures = {c: pool.submit(timed, c, examples) for c in CONDITIONS}
        for c, f in futures.items():
            try:
                results[c] = f.result()
            except BaseException as e:  # API 키 없음(sys.exit) 포함
                errors[c] = ("API 키 없음 (.env의 GOOGLE_API_KEY 확인)" if isinstance(e, SystemExit)
                             else f"{type(e).__name__}: {e}")
    return results, errors


# 파이프라인 연결 전/API 한도 소진 시 화면 확인용 가짜 결과
DUMMY_SOURCE = "요즘 AI가 만드는 글이 다 비슷비슷해지고 있다. 개인화 번역은 사람마다 자기 말투로 번역되게 해서 이 문제를 줄이려는 방법이다."
DUMMY_OUTPUTS = {
    "A": "Moreover, AI-generated content is becoming increasingly homogenous. "
         "Personalized translation plays a crucial role in fostering a more seamless and "
         "authentic voice for each individual.",
    "B": "These days, AI writing is getting really similar. Personalized translation "
         "tries to fix this by translating in each person's own way of speaking.",
    "C": "AI writing all sounds the same these days. Personalized translation tries to "
         "fix that, so each person's text comes out in their own voice.",
}


# ==================================================
# 한→한 문체 변환 (src/ko_style.py, D-050)
#   중립적인 한국어를 선택한 사람의 문체로 다시 씁니다. 생성만 하고 의미 검증은 하지 않습니다.
#   RAG 예시는 전체 코퍼스에서 검색합니다(평가 실험 scripts/ko_style_transfer.py와 달리 평가 문서 제외 없음).
# ==================================================
TASK_EN = "한→영 번역"
TASK_KO = "한→한 문체 변환"
KO_MODES = {
    "P+R": "P+R · 문체 지시 + 본인 글 예시",
    "P": "P · 문체 지시만",
    "R": "R · 본인 글 예시만",
}
DUMMY_KO_SOURCE = ("최근 AI가 쓴 글은 서로 비슷해지는 경향이 있습니다. 그러나 사람마다 글을 쓰는 방식은 다릅니다. "
                   "이 차이는 말투에 그대로 드러나므로 번역에서도 지켜져야 합니다. 따라서 개인의 문체를 반영하는 변환이 필요합니다.")
DUMMY_KO_OUTPUTS = {  # 가짜 결과(화면 확인용): 실제 모델 출력이 아님
    "ganghee": "최근 AI가 쓴 글은 서로 비슷해지는 경향이 있다. 그러나 사람마다 글을 쓰는 방식은 다르며, "
               "이 차이는 말투에 그대로 드러나므로 번역에서도 지켜져야 한다. 그래서 개인의 문체를 반영하는 변환이 필요하다.",
    "nakyung": "최근 AI가 쓴 글은 서로 비슷해지는 경향이 있다. 그러나 사람마다 글을 쓰는 방식은 다르다. "
               "이 차이는 말투에 그대로 드러나기에 번역에서도 지켜져야 한다. 따라서 개인의 문체를 반영하는 변환이 필요하다.",
    "suyoung": "최근 AI가 쓴 글은 서로 비슷해지는 경향이 있지만 사람마다 글을 쓰는 방식은 다르고, "
               "이 차이는 말투에 그대로 드러나는 것이어서 번역에서도 지켜져야 하며, 개인의 문체를 반영하는 변환이 필요하다.",
}


@st.cache_resource(show_spinner=False)
def warm_up_ko():
    """Kiwi를 미리 불러 둔다(한→한 모드의 기능 감지·문체 판정용)."""
    from src import style_features as sf
    sf._get_kiwi().tokenize("준비")
    return True


def run_ko_style(source, user, mode, use_dummy):
    from src import ko_style
    start = time.time()
    dummy = None
    if use_dummy:
        dummy = DUMMY_KO_OUTPUTS[user] if source.strip() == DUMMY_KO_SOURCE else "[가짜 결과] " + source
    r = ko_style.convert_style(source, user, mode, dummy_output=dummy)
    r["seconds"] = time.time() - start
    return r


def render_ko_style(persona, use_dummy, mode):
    from src import config, ko_style
    user = PERSONAS[persona]
    st.caption("AI가 쓴 듯한 평범한 한국어를 선택한 사람의 문체로 다시 씁니다. 생성만 하고 의미 검증은 하지 않으니, "
               "변환문을 읽고 뜻이 같은지 직접 확인하세요.")
    if config.CORPUS_SPLIT and not use_dummy:
        st.warning(f"LINGUA_CORPUS_SPLIT={config.CORPUS_SPLIT}이 설정돼 있어 일부 글을 뺀 코퍼스로 검색합니다. "
                   "데모는 전체 코퍼스를 써야 하니 이 환경변수를 지우고 다시 실행하세요.")

    source = st.text_area("바꿀 한국어 글 (평범한 말투)", value=DUMMY_KO_SOURCE, height=200)
    n_chars = len(source.strip())
    st.caption(f"{n_chars:,}자")
    if 0 < n_chars < ko_style.MIN_CHARS_FOR_STYLE_SCORE:
        st.info(f"{ko_style.SHORT_INPUT_NOTICE} (1,000자 미만 · 문체 판정 점수는 1,000자 이상일 때만 표시합니다)")

    if st.button("문체 변환 실행", type="primary", width="stretch", disabled=n_chars == 0):
        try:
            if not use_dummy:
                warm_up_ko()
            with st.spinner(f"{persona} 문체로 변환 중..."):
                r = run_ko_style(source, user, mode, use_dummy)
            r.update(persona=persona, source=source, dummy=use_dummy)
            st.session_state["ko_result"] = r
        except BaseException as e:  # API 키 없음(sys.exit) 포함
            st.error("API 키 없음 (.env의 GOOGLE_API_KEY 확인)  — 사이드바에서 가짜 결과를 켜면 화면은 확인할 수 있어요"
                     if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}")

    r = st.session_state.get("ko_result")
    if not r:
        st.info("글을 입력하고 버튼을 눌러 주세요.")
        return

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("입력")
        st.text_area("입력", r["source"], height=260, disabled=True, label_visibility="collapsed")
    with right:
        st.subheader(f"{r['persona']} 문체로 변환")
        st.text_area("변환문", r["converted"], height=260, disabled=True, label_visibility="collapsed")
        st.caption(("가짜 결과 · " if r["dummy"] else "") + f"{KO_MODES[r['mode']]} · {r['seconds']:.1f}초")

    st.markdown("**적용된 문체 지시**")
    if "P" not in r["mode"]:
        st.caption("이 변환 방식은 문체 지시를 쓰지 않습니다.")
    elif r["lines"]:
        for line in r["lines"]:
            st.markdown(f"- {line}")
    else:
        st.caption("이번 입력에는 적용된 문체 지시가 없습니다 (해당 표현이 입력에 없어서, 지시는 입력에 그 기능이 있을 때만 들어갑니다).")
    with st.expander("항상 들어가는 공통 규칙"):
        for i, rule in enumerate(ko_style.COMMON_RULES, 1):
            st.markdown(f"{i}. {rule}")

    if "R" in r["mode"]:
        with st.expander(f"사용한 예시 {len(r['examples'])}개 ({r['persona']} 본인 글, 전체 코퍼스에서 검색)"):
            for i, e in enumerate(r["examples"], 1):
                st.markdown(f"**예시{i}** · {e.get('type') or ''} · 유사도 {e['score']:.2f}")
                st.write(e["text"])

    st.markdown("**문체 판정**")
    if len(r["source"].strip()) < ko_style.MIN_CHARS_FOR_STYLE_SCORE:
        st.caption(f"문체 판정 점수는 1,000자 이상일 때만 표시합니다 (현재 {len(r['source'].strip()):,}자).")
        return
    try:
        warm_up_ko()
        with st.spinner("문체 판정 계산 중... (처음 한 번은 기준 계산에 시간이 걸려요)"):
            before = ko_style.style_judgment(r["source"])
            after = ko_style.style_judgment(r["converted"])
    except BaseException as e:
        st.warning(f"문체 판정 계산 실패 ({type(e).__name__}: {e})")
        return
    names = {v: k for k, v in PERSONAS.items()}
    df = pd.DataFrame({"입력": before["distances"], "변환문": after["distances"]}).rename(index=names).round(3)
    st.dataframe(df, width="stretch")
    st.caption(f"기능 형태소 Delta 거리(작을수록 그 사람 글에 가까움). 입력은 {names[before['nearest']]}, "
               f"변환문은 {names[after['nearest']]} 쪽에 가장 가깝습니다. 지시에 쓴 표현(그러나·따라서·므로 등)은 뺀 값이며, "
               "글 한 편(1,000자 이상) 기준의 참고값입니다. 소표본이라 방향만 보세요.")


# ==================================================
# AI 클리셰 사전 (팀 공용: src/cliches.py, evaluate_metrics.py와 같은 사전)
# ==================================================
from src.cliches import ENGLISH_AI_CLICHES  # noqa: E402

ALL_AI_CLICHES = sorted({p for ps in ENGLISH_AI_CLICHES.values() for p in ps},
                        key=len, reverse=True)   # 긴 표현부터 매칭
# 단어 경계(\b)로 매칭: "vital"이 "revitalize" 안에서 잡히는 오류 방지
CLICHE_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, ALL_AI_CLICHES)) + r")\b",
                       re.IGNORECASE)


# ==================================================
# 지표 계산 (nltk 없이 동작)
# ==================================================
def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())


def find_cliches(text):
    return [m.group().lower() for m in CLICHE_RE.finditer(str(text))]


def cliche_rate(text):
    """상투어구가 차지하는 단어 비율(%)"""
    words = tokenize(text)
    if not words:
        return 0.0
    cliche_words = sum(len(tokenize(c)) for c in find_cliches(text))
    return 100.0 * cliche_words / len(words)


def lexical_diversity(text):
    words = tokenize(text)
    return len(set(words)) / len(words) if words else 0.0


def avg_sentence_length(text):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if tokenize(s)]
    return sum(len(tokenize(s)) for s in sents) / len(sents) if sents else 0.0


def length_match(text, ref_len):
    """사용자 평균 문장 길이와 얼마나 비슷한지 (1에 가까울수록 비슷)"""
    if not ref_len:
        return 0.0
    return max(0.0, 1 - abs(avg_sentence_length(text) - ref_len) / ref_len)


@st.cache_data
def load_corpus(persona):
    """data/eval/english_<user>.jsonl 의 text를 모두 이어 붙인 실제 영어 글"""
    path = EVAL_DIR / f"english_{PERSONAS[persona]}.jsonl"
    if not path.exists():
        return ""
    with open(path, encoding="utf-8") as f:
        return "\n".join(json.loads(line)["text"] for line in f if line.strip())


# ==================================================
# C 검증 결과 패널
# ==================================================
SEVERITY_KO = {"critical": "심각", "major": "주의", "minor": "참고"}
VERDICT_KO = {"PASS": "통과", "REVIEW": "검토 필요", "FAIL": "의미 오류"}


def render_verification(detail):
    if not detail:
        return
    verdict = detail["verdict"]
    head = f"C 의미 검증: {VERDICT_KO[verdict]} ({verdict}) · Meaning Score {detail['meaning_score']:.0f}/100"
    with st.expander(head, expanded=verdict != "PASS"):
        comp = detail["meaning_components"]
        sim = comp.get("semantic_similarity")
        st.caption(
            f"원문 claim {comp['source_claim_count']}개 · 보존도 {comp['fidelity']:.2f} · "
            f"추가 내용 감점 ×{comp['addition_factor']:.2f} · 규칙 검사 ×{comp['rule_factor']:.2f}"
            + (f" · 상한 {comp['severity_cap']:.0f}점" if comp.get("severity_cap", 100) < 100 else "")
            + (f" · 임베딩 유사도 {sim:.2f}" if sim is not None else "")
            + f" · 판정 모델 {detail['judge_model']}"
        )
        issues = detail["issues"]
        if issues:
            for i in issues:
                where = f" — “{i['evidence']}”" if i.get("evidence") else ""
                st.markdown(f"- **[{SEVERITY_KO[i['severity']]}] {i['label_ko']}**{where}  \n  {i.get('note', '')}")
        else:
            st.markdown("찾은 의미 문제 없음.")
        history = detail["verification_history"]
        if len(history) > 1:
            st.markdown("**시도별 결과**")
            for n, (t, v) in enumerate(zip(detail["translations"], history), 1):
                mark = " ← 채택" if n == detail["best_attempt"] else ""
                st.markdown(f"{n}. {v['verdict']} · {v['meaning_score']:.0f}점{mark}  \n  {t}")


# ==================================================
# 화면
# ==================================================
def main():
    st.title("Lingua Me")
    with st.sidebar:
        st.header("설정")
        task = st.radio("작업", [TASK_EN, TASK_KO])
        persona = st.radio("누구 말투로 번역할까요?" if task == TASK_EN else "누구 문체로 바꿀까요?", list(PERSONAS))
        ko_mode = "P+R"
        if task == TASK_KO:
            ko_mode = st.radio("변환 방식", list(KO_MODES), format_func=KO_MODES.get)
        use_dummy = st.toggle("데모용 가짜 결과 (API 호출 안 함)", value=False,
                              help="API 키 없이 화면만 확인할 때")
        if task == TASK_EN:
            corpus = load_corpus(persona)
            if corpus:
                st.caption(f"{persona}의 실제 영어 글 {len(tokenize(corpus))}단어를 기준으로 비교합니다.")
            else:
                st.warning(f"data/eval/english_{PERSONAS[persona]}.jsonl 이 없어 스타일 기준 비교를 건너뜁니다.")

    if task == TASK_KO:
        render_ko_style(persona, use_dummy, ko_mode)
        return
    st.caption("한국어 → 영어 개인화 번역 비교 및 AI 획일화 정량 지표 대시보드")

    source = st.text_area("한국어 원문", value=DUMMY_SOURCE, height=120)

    if st.button("번역 실행 및 지표 분석", type="primary", width="stretch"):
        if not use_dummy:
            try:
                warm_up(PERSONAS[persona])
            except BaseException as e:
                st.warning(f"준비 단계 실패({type(e).__name__}: {e}) — 번역을 그대로 시도합니다.")
        with st.spinner("A·B·C 동시 번역 중... (C는 검증까지 하므로 가장 오래 걸려요)"):
            results, errors = run_all_conditions(source, PERSONAS[persona], use_dummy)
        for c, msg in errors.items():
            st.error(f"{CONDITIONS[c]} 실패 — {msg}  (사이드바에서 가짜 결과를 켜면 화면은 확인할 수 있어요)")
        if len(results) == len(CONDITIONS):
            st.session_state["outputs"] = {c: r["text"] for c, r in results.items()}
            st.session_state["infos"] = {c: r["info"] for c, r in results.items()}
            st.session_state["detail_c"] = results["C"].get("detail")
            st.session_state["persona"] = persona

    outputs = st.session_state.get("outputs")
    if not outputs:
        st.info("원문을 입력하고 버튼을 눌러 주세요.")
        return
    persona = st.session_state["persona"]
    corpus = load_corpus(persona)

    st.divider()
    cols = st.columns(3)
    for col, (c, label) in zip(cols, CONDITIONS.items()):
        with col:
            st.subheader(label)
            st.text_area(label, outputs[c], height=180, disabled=True,
                         label_visibility="collapsed")
            st.caption(st.session_state.get("infos", {}).get(c, ""))

    render_verification(st.session_state.get("detail_c"))

    # ---------------- 지표 표
    ref_len = avg_sentence_length(corpus) if corpus else 0.0
    rows = []
    for c, label in CONDITIONS.items():
        t = outputs[c]
        rows.append({
            "조건": label,
            "AI 클리셰 비율(%)": cliche_rate(t),
            "어휘 다양도(TTR)": lexical_diversity(t),
            "평균 문장 길이": avg_sentence_length(t),
            "문장 길이 일치도": length_match(t, ref_len) if corpus else None,
        })
    if corpus:
        rows.append({
            "조건": f"참고 · {persona} 실제 글",
            "AI 클리셰 비율(%)": cliche_rate(corpus),
            "어휘 다양도(TTR)": None,   # 길이가 달라 TTR 직접 비교는 부적절
            "평균 문장 길이": ref_len,
            "문장 길이 일치도": 1.0,
        })
    df = pd.DataFrame(rows)

    st.header("AI 획일화 및 스타일 지표")
    k1, k2, k3 = st.columns(3)
    k1.metric("C 클리셰 비율", f"{df.iloc[2]['AI 클리셰 비율(%)']:.1f}%",
              f"{df.iloc[2]['AI 클리셰 비율(%)'] - df.iloc[0]['AI 클리셰 비율(%)']:+.1f}%p vs A",
              delta_color="inverse")
    k2.metric("C 어휘 다양도", f"{df.iloc[2]['어휘 다양도(TTR)']:.2f}",
              f"{df.iloc[2]['어휘 다양도(TTR)'] - df.iloc[0]['어휘 다양도(TTR)']:+.2f} vs A")
    if corpus:
        k3.metric(f"C 문장 길이 일치도 ({persona} 기준)", f"{df.iloc[2]['문장 길이 일치도']:.2f}",
                  f"{df.iloc[2]['문장 길이 일치도'] - df.iloc[0]['문장 길이 일치도']:+.2f} vs A")

    tab1, tab2, tab3 = st.tabs(["지표 시각화", "데이터 상세 표", "AI 클리셰 분석"])

    with tab1:
        colors = {CONDITIONS["A"]: "#B0BEC5", CONDITIONS["B"]: "#90CAF9",
                  CONDITIONS["C"]: "#1565C0", f"참고 · {persona} 실제 글": "#FFB74D"}
        g1, g2 = st.columns(2)
        with g1:
            fig = px.bar(df, x="조건", y="AI 클리셰 비율(%)", color="조건",
                         color_discrete_map=colors, text_auto=".1f",
                         title="AI 클리셰 비율 (낮을수록 좋음)")
            fig.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig, width="stretch")
        with g2:
            if corpus:
                fig = px.bar(df, x="조건", y="평균 문장 길이", color="조건",
                             color_discrete_map=colors, text_auto=".1f",
                             title=f"평균 문장 길이 ({persona} 실제 글에 가까울수록 좋음)")
                fig.update_layout(showlegend=False, xaxis_title="")
                st.plotly_chart(fig, width="stretch")

    with tab2:
        st.dataframe(df.set_index("조건").round(3), width="stretch")

    with tab3:
        for c, label in CONDITIONS.items():
            found = find_cliches(outputs[c])
            st.markdown(f"**{label}**: " + (", ".join(sorted(set(found))) if found else "탐지된 클리셰 없음"))
        with st.expander("현재 사용 중인 클리셰 사전"):
            for category, phrases in ENGLISH_AI_CLICHES.items():
                st.markdown(f"**{category}**: " + ", ".join(phrases))


if __name__ == "__main__":
    main()
