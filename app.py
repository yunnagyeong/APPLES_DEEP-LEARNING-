import json
import re
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Personalized Translation & AI Uniformity Benchmark",
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
    st.title("Personalized Translation & AI Uniformity Benchmark")
    st.caption("한국어 → 영어 개인화 번역 비교 및 AI 획일화 정량 지표 대시보드")

    with st.sidebar:
        st.header("설정")
        persona = st.radio("누구 말투로 번역할까요?", list(PERSONAS))
        use_dummy = st.toggle("데모용 가짜 결과 (API 호출 안 함)", value=False,
                              help="API 키 없이 화면만 확인할 때")
        corpus = load_corpus(persona)
        if corpus:
            st.caption(f"{persona}의 실제 영어 글 {len(tokenize(corpus))}단어를 기준으로 비교합니다.")
        else:
            st.warning(f"data/eval/english_{PERSONAS[persona]}.jsonl 이 없어 스타일 기준 비교를 건너뜁니다.")

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
