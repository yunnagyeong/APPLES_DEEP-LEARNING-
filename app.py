import streamlit as st
import pandas as pd
import re
from kiwipiepy import Kiwi

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Personalized Translation & Style Profile Benchmark",
    layout="wide"
)

# ---------------------------------------------------------
# Kiwi 형태소 분석기 초기화 (캐싱 처리로 속도 최적화)
# ---------------------------------------------------------
@st.cache_resource
def get_kiwi():
    return Kiwi()

kiwi = get_kiwi()

# ---------------------------------------------------------
# 팀 명세서 사전 정의 (Dictionary)
# ---------------------------------------------------------
LOGICAL_CONNECTIVES = [
    "그러므로", "따라서", "하지만", "그러나", "왜냐하면",
    "즉", "반면에", "그리고", "결과적으로", "그럼에도"
]

HEDGING_PATTERNS = [
    "것 같다", "수 있다", "수도 있다", "일 수 있다",
    "아마", "대체로", "듯하다", "볼 수 있다"
]

EXAMPLE_MARKERS = ["예를 들어", "가령", "예컨대"]
CONCLUSION_MARKERS = ["결론적으로", "요컨대", "따라서", "결국"]

FORMAL_ENDINGS = re.compile(r'(습니다|합니다|입니다|됩니다)[\.\?!]?$')
DIRECT_ENDINGS = re.compile(r'(이다|하다|한다|된다|있다|없다)[\.\?!]?$')

# ---------------------------------------------------------
# StyleProfile 명세서 기반 스타일/획일화 지표 계산 함수
# ---------------------------------------------------------
def analyze_style_profile(text: str) -> dict:
    """팀원①의 StyleProfile 명세 초안(9개 지표)을 계산하는 엔진"""
    if not text.strip():
        return {
            "sentence_length_avg": 0.0,
            "formality_score": 0.0,
            "directness_score": 0.0,
            "logical_connective_rate": 0.0,
            "hedging_rate": 0.0,
            "conclusion_position": "none",
            "example_usage_rate": 0.0,
            "lexical_diversity": 0.0,
            "paragraph_length_avg": 0.0,
            "source_example_count": 0
        }

    # 문단 분리
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    num_paragraphs = max(len(paragraphs), 1)

    # Kiwi 문장 분리
    kiwi_sentences = kiwi.split_into_sents(text)
    sentences = [s.text.strip() for s in kiwi_sentences if s.text.strip()]
    num_sentences = max(len(sentences), 1)

    # 1. sentence_length_avg: 평균 문장 길이 (어절 수)
    sentence_lengths = [len(s.split()) for s in sentences]
    sentence_length_avg = sum(sentence_lengths) / num_sentences

    # 2. formality_score: 격식체 비율
    formal_count = sum(1 for s in sentences if FORMAL_ENDINGS.search(s))
    formality_score = formal_count / num_sentences

    # 3. directness_score: 단정형 어미 비율
    direct_count = sum(1 for s in sentences if DIRECT_ENDINGS.search(s))
    directness_score = direct_count / num_sentences

    # 4. logical_connective_rate: 문장당 접속 표현 개수
    conn_count = sum(text.count(conn) for conn in LOGICAL_CONNECTIVES)
    logical_connective_rate = conn_count / num_sentences

    # 5. hedging_rate: 문장당 hedging 표현 개수
    hedge_count = sum(text.count(h) for h in HEDGING_PATTERNS)
    hedging_rate = hedge_count / num_sentences

    # 6. conclusion_position: 결론 표지어 위치 ("front" | "back" | "mixed")
    first_conclusion_para = -1
    for p_idx, p in enumerate(paragraphs):
        if any(marker in p for marker in CONCLUSION_MARKERS):
            first_conclusion_para = p_idx
            break
    
    if first_conclusion_para == -1:
        conclusion_position = "none"
    elif first_conclusion_para <= num_paragraphs / 3:
        conclusion_position = "front"
    elif first_conclusion_para >= (2 * num_paragraphs) / 3:
        conclusion_position = "back"
    else:
        conclusion_position = "mixed"

    # 7. example_usage_rate: 문장당 예시 표지어 개수
    example_count = sum(text.count(ex) for ex in EXAMPLE_MARKERS)
    example_usage_rate = example_count / num_sentences

    # 8. lexical_diversity: Kiwi 형태소 기반 TTR (7주차 AI 획일화 측정 지표)
    tokens = [t.form for t in kiwi.tokenize(text)]
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    lexical_diversity = (unique_tokens / total_tokens) if total_tokens > 0 else 0.0

    # 9. paragraph_length_avg: 문단당 평균 문장 수
    paragraph_length_avg = num_sentences / num_paragraphs

    return {
        "sentence_length_avg": round(sentence_length_avg, 2),
        "formality_score": round(formality_score, 2),
        "directness_score": round(directness_score, 2),
        "logical_connective_rate": round(logical_connective_rate, 2),
        "hedging_rate": round(hedging_rate, 2),
        "conclusion_position": conclusion_position,
        "example_usage_rate": round(example_usage_rate, 2),
        "lexical_diversity": round(lexical_diversity, 3),
        "paragraph_length_avg": round(paragraph_length_avg, 2),
        "source_example_count": 1
    }

# ---------------------------------------------------------
# Sidebar: 데모 데이터 프리셋
# ---------------------------------------------------------
with st.sidebar:
    st.header("데모 및 설정")
    st.info("팀원①의 `StyleProfile` 명세 스키마(9개 지표)와 Kiwi 형태소 분석기가 연동되어 있습니다.")
    
    preset = st.selectbox(
        "테스트 텍스트 프리셋",
        ["직접 입력", "비즈니스 보고서 스타일", "자연스러운 구어/블로그 스타일"]
    )
    
    default_src, default_std, default_pers = "", "", ""
    if preset == "비즈니스 보고서 스타일":
        default_src = "Artificial intelligence models often produce repetitive and monotonous phrases. Therefore, personalized adaptation is required."
        default_std = "인공지능 모델은 종종 반복적이고 단조로운 문구를 생성합니다. 그러므로 개인화된 적응이 필요합니다."
        default_pers = "최근 생성형 AI는 표현의 획일화 경향을 보입니다. 따라서 사용자의 어휘 특성을 반영하는 개인화 튜닝이 필수적입니다. 예를 들어 맞춤형 번역 모델을 구축할 수 있습니다."
    elif preset == "자연스러운 구어/블로그 스타일":
        default_src = "I was worried about falling behind, but taking a good rest might actually help in the long run."
        default_std = "나는 뒤처지는 것에 대해 걱정했지만, 장기적으로는 푹 쉬는 것이 실제로 도움이 될 수도 있습니다."
        default_pers = "진도가 밀려서 좀 불안했는데, 길게 보면 오히려 푹 쉰 게 약이 된 것 같다. 내일부터 다시 해보려고 한다."

# ---------------------------------------------------------
# UI 메인 헤더
# ---------------------------------------------------------
st.title("3-Column Translation & Style Profile Benchmark")
st.caption("팀 인터페이스 규격(Kiwi 기반 StyleProfile 9대 지표 및 AI 획일화 TTR) 실시간 비교 데모")

st.markdown("---")

# ---------------------------------------------------------
# 3열 레이아웃: 원문 | 일반 번역 | Personalized Translation
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Source Text (원문)")
    src_text = st.text_area("Source Text", value=default_src, height=220, placeholder="원문을 입력하세요...", label_visibility="collapsed")

with col2:
    st.subheader("2. Standard Translation (일반 번역)")
    std_text = st.text_area("Standard Translation", value=default_std, height=220, placeholder="일반 번역 결과...", label_visibility="collapsed")

with col3:
    st.subheader("3. Personalized Translation (개인화 번역)")
    pers_text = st.text_area("Personalized Translation", value=default_pers, height=220, placeholder="개인화 번역 결과...", label_visibility="collapsed")

# ---------------------------------------------------------
# 지표 분석 영역 (팀원① StyleProfile 스키마 기준)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Style Profile & AI 획일화(Diversity) 비교 분석")

if st.button("스타일 지표 및 획일화 분석 실행", type="primary"):
    if not std_text.strip() or not pers_text.strip():
        st.warning("분석할 일반 번역문과 개인화 번역문을 모두 입력해주세요.")
    else:
        profile_std = analyze_style_profile(std_text)
        profile_pers = analyze_style_profile(pers_text)

        # 1. 핵심 지표 요약 카드
        k1, k2, k3, k4 = st.columns(4)
        
        diff_ttr = profile_pers["lexical_diversity"] - profile_std["lexical_diversity"]
        diff_hedge = profile_pers["hedging_rate"] - profile_std["hedging_rate"]
        
        k1.metric("표준 번역 어휘 다양성 (TTR)", f"{profile_std['lexical_diversity']:.3f}")
        k2.metric("개인화 어휘 다양성 (TTR)", f"{profile_pers['lexical_diversity']:.3f}", delta=f"{diff_ttr:+.3f}")
        k3.metric("격식성 (Formality)", f"{profile_pers['formality_score']:.2f}")
        k4.metric("Hedging 빈도", f"{profile_pers['hedging_rate']:.2f}", delta=f"{diff_hedge:+.2f}")

        # 2. 팀원①의 StyleProfile JSON 규격 테이블
        st.write("##### `StyleProfile` 스키마 전체 항목 비교")
        df_display = pd.DataFrame([profile_std, profile_pers], index=["일반 번역 (Standard)", "개인화 번역 (Personalized)"])
        st.dataframe(df_display, use_container_width=True)

        # 3. 레이더/막대 차트로 정량 지표 비교
        st.write("##### 주요 문체 정량 수치 비교")
        chart_metrics = [
            "sentence_length_avg", 
            "formality_score", 
            "directness_score", 
            "logical_connective_rate", 
            "hedging_rate", 
            "example_usage_rate", 
            "lexical_diversity"
        ]
        
        chart_df = pd.DataFrame({
            "Standard": [profile_std[m] for m in chart_metrics],
            "Personalized": [profile_pers[m] for m in chart_metrics]
        }, index=chart_metrics)

        st.bar_chart(chart_df)