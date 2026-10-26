import streamlit as st
import pandas as pd
import plotly.express as px
from nltk.tokenize import word_tokenize
from collections import Counter


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Personalized Translation & Style Profile Benchmark",
    layout="wide"
)


# ==================================================
# AI Cliche Lexicon
# ==================================================

ENGLISH_AI_CLICHES = {

    "transitions": [
        "furthermore",
        "moreover",
        "in addition",
        "additionally",
        "notably",
        "consequently",
        "subsequently",
        "it is worth noting that",
        "it is noteworthy that",
        "importantly",
        "in summary",
        "to summarize",
        "in conclusion",
        "all in all",
        "ultimately",
        "moving forward"
    ],

    "hedging_and_boosters": [
        "plays a crucial role",
        "vital to consider",
        "delve into",
        "testament to",
        "foster",
        "pivotal",
        "beacon"
    ],

    "elevated_modifiers": [
        "pivotal",
        "crucial",
        "vital",
        "paramount",
        "essential",
        "indispensable",
        "multifaceted",
        "nuanced",
        "intricate",
        "meticulous",
        "comprehensive",
        "robust",
        "dynamic",
        "vibrant",
        "seamless",
        "transformative",
        "groundbreaking"
    ]
}


# ==================================================
# 모든 AI Cliche 표현 하나의 리스트로 만들기
# ==================================================

def get_all_cliches():

    cliches = []

    for category in ENGLISH_AI_CLICHES.values():
        cliches.extend(category)

    # 중복 제거
    return list(set(cliches))


ALL_AI_CLICHES = get_all_cliches()


# ==================================================
# AI 획일화 지표 ①
# AI Cliche Count
# ==================================================

def calculate_cliche_count(text):

    if not text or not text.strip():
        return 0

    text_lower = text.lower()

    count = 0

    for phrase in ALL_AI_CLICHES:
        count += text_lower.count(
            phrase.lower()
        )

    return count


# ==================================================
# AI 획일화 지표 ②
# AI Cliche Rate
# ==================================================

def calculate_cliche_rate(text):

    if not text or not text.strip():
        return 0.0

    words = word_tokenize(
        text.lower()
    )

    # 영어 단어만 사용
    words = [
        word
        for word in words
        if word.isalpha()
    ]

    if len(words) == 0:
        return 0.0

    cliche_count = calculate_cliche_count(
        text
    )

    return cliche_count / len(words)


# ==================================================
# AI 획일화 지표 ③
# Lexical Diversity
# ==================================================

def calculate_lexical_diversity(text):

    if not text or not text.strip():
        return 0.0

    words = word_tokenize(
        text.lower()
    )

    words = [
        word
        for word in words
        if word.isalpha()
    ]

    if len(words) == 0:
        return 0.0

    unique_words = len(
        set(words)
    )

    return unique_words / len(words)


# ==================================================
# 어떤 클리셰가 등장했는지 확인
# ==================================================

def find_cliches(text):

    if not text:
        return []

    text_lower = text.lower()

    found = []

    for phrase in ALL_AI_CLICHES:

        if phrase.lower() in text_lower:

            found.append(phrase)

    return found


# ==================================================
# Dummy Generation Data
# ==================================================

DUMMY_GENERATION_RESULT = {

    # 영어 원문
    "source_text":
        "AI models are increasingly generating homogenous content. "
        "Personalized translation approaches aim to alleviate this "
        "uniformity by adapting outputs to individual stylistic signatures.",

    # 일반 번역
    "standard_translation":
        "AI 모델은 점차 획일화된 콘텐츠를 생성하고 있습니다. "
        "개인화된 번역 접근 방식은 출력을 개별 스타일 서명에 맞게 "
        "조정하여 이러한 획일성을 완화하는 것을 목표로 합니다.",

    # 개인화 번역
    "personalized_translation":
        "생성형 AI 모델이 내놓는 결과물이 갈수록 비슷비슷해지고 있죠. "
        "맞춤형 번역 기법은 사용자의 고유한 어조와 문체를 반영함으로써 "
        "이러한 획일화 현상을 효과적으로 해소합니다."
}


# ==================================================
# 비교용 영어 텍스트
#
# 실제 Generation 모듈 연결 전까지 테스트용
# ==================================================

DUMMY_STANDARD_ENGLISH = (
    "AI models are increasingly generating homogenous content. "
    "Furthermore, it is important to note that these models "
    "play a crucial role in shaping modern communication. "
    "Moreover, this phenomenon is particularly significant."
)


DUMMY_PERSONALIZED_ENGLISH = (
    "AI models are producing increasingly similar content. "
    "These outputs can make different writers sound alike. "
    "A personalized approach instead adapts the wording "
    "to the writer's own style."
)


# ==================================================
# 기존 Verification Dummy Data
# ==================================================

DUMMY_VERIFICATION_RESULT = {

    "sentence_length_avg": {
        "Standard": 5.5,
        "Personalized": 7.67
    },

    "formality_score": {
        "Standard": 1.0,
        "Personalized": 1.0
    },

    "directness_score": {
        "Standard": 0.0,
        "Personalized": 0.0
    },

    "logical_connective_rate": {
        "Standard": 0.50,
        "Personalized": 0.33
    },

    "hedging_rate": {
        "Standard": 0.0,
        "Personalized": 0.0
    },

    "example_usage_rate": {
        "Standard": 0.0,
        "Personalized": 0.0
    }
}


# ==================================================
# Main UI
# ==================================================

def main():

    st.title(
        "Personalized Translation & AI Uniformity Benchmark"
    )

    st.caption(
        "번역 비교 및 AI 획일화 정량 지표 분석 대시보드"
    )

    st.divider()


    # ==================================================
    # 1. 3열 번역 비교
    # ==================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.subheader("원문")

        source_input = st.text_area(
            "원문",
            value=DUMMY_GENERATION_RESULT["source_text"],
            height=200,
            label_visibility="collapsed"
        )


    with col2:

        st.subheader("일반 번역")

        standard_translation = st.text_area(
            "일반 번역",
            value=DUMMY_GENERATION_RESULT[
                "standard_translation"
            ],
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )


    with col3:

        st.subheader("Personalized Translation")

        personalized_translation = st.text_area(
            "개인화 번역",
            value=DUMMY_GENERATION_RESULT[
                "personalized_translation"
            ],
            height=200,
            disabled=True,
            label_visibility="collapsed"
        )


    # ==================================================
    # 2. 실행 버튼
    # ==================================================

    st.write("")

    run_button = st.button(
        "번역 지표 측정 및 AI 획일화 분석 실행",
        type="primary",
        use_container_width=True
    )

    st.divider()


    # ==================================================
    # 3. 실행
    # ==================================================

    if run_button:

        st.header(
            "AI 획일화 정량 지표"
        )


        # ==================================================
        # 실제 AI 획일화 계산
        # ==================================================

        standard_cliche_count = (
            calculate_cliche_count(
                DUMMY_STANDARD_ENGLISH
            )
        )

        personalized_cliche_count = (
            calculate_cliche_count(
                DUMMY_PERSONALIZED_ENGLISH
            )
        )


        standard_cliche_rate = (
            calculate_cliche_rate(
                DUMMY_STANDARD_ENGLISH
            )
        )

        personalized_cliche_rate = (
            calculate_cliche_rate(
                DUMMY_PERSONALIZED_ENGLISH
            )
        )


        standard_diversity = (
            calculate_lexical_diversity(
                DUMMY_STANDARD_ENGLISH
            )
        )

        personalized_diversity = (
            calculate_lexical_diversity(
                DUMMY_PERSONALIZED_ENGLISH
            )
        )


        # ==================================================
        # 발견된 클리셰
        # ==================================================

        standard_found = find_cliches(
            DUMMY_STANDARD_ENGLISH
        )

        personalized_found = find_cliches(
            DUMMY_PERSONALIZED_ENGLISH
        )


        # ==================================================
        # 핵심 KPI
        # ==================================================

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)


        with kpi1:

            st.metric(
                "Standard Cliche",
                standard_cliche_count
            )


        with kpi2:

            st.metric(
                "Personalized Cliche",
                personalized_cliche_count
            )


        with kpi3:

            st.metric(
                "Standard Lexical Diversity",
                f"{standard_diversity:.2f}"
            )


        with kpi4:

            st.metric(
                "Personalized Lexical Diversity",
                f"{personalized_diversity:.2f}"
            )


        # ==================================================
        # AI 획일화 데이터
        # ==================================================

        uniformity_df = pd.DataFrame({

            "Metric": [
                "AI Cliche Rate",
                "Lexical Diversity"
            ],

            "Standard": [
                standard_cliche_rate,
                standard_diversity
            ],

            "Personalized": [
                personalized_cliche_rate,
                personalized_diversity
            ]
        })


        # ==================================================
        # 기존 Verification + AI 획일화
        # ==================================================

        metrics_data = []


        for metric_name, values in (
            DUMMY_VERIFICATION_RESULT.items()
        ):

            metrics_data.append({

                "Metric": metric_name,

                "Standard":
                    values.get(
                        "Standard",
                        0.0
                    ),

                "Personalized":
                    values.get(
                        "Personalized",
                        0.0
                    )
            })


        metrics_data.append({

            "Metric":
                "AI Cliche Rate",

            "Standard":
                standard_cliche_rate,

            "Personalized":
                personalized_cliche_rate
        })


        metrics_data.append({

            "Metric":
                "Lexical Diversity",

            "Standard":
                standard_diversity,

            "Personalized":
                personalized_diversity
        })


        df = pd.DataFrame(
            metrics_data
        )


        # ==================================================
        # 탭
        # ==================================================

        tab1, tab2, tab3 = st.tabs([
            "지표 시각화",
            "데이터 상세 표",
            "AI 클리셰 분석"
        ])


        # ==================================================
        # TAB 1
        # ==================================================

        with tab1:

            st.markdown(
                "##### AI 획일화 및 스타일 지표"
            )


            df_long = pd.melt(

                df,

                id_vars=["Metric"],

                value_vars=[
                    "Standard",
                    "Personalized"
                ],

                var_name="Type",

                value_name="Score"
            )


            # 문장 길이 제외
            df_rates = df_long[
                df_long["Metric"]
                != "sentence_length_avg"
            ]


            df_length = df_long[
                df_long["Metric"]
                == "sentence_length_avg"
            ]


            chart_col1, chart_col2 = (
                st.columns([2, 1])
            )


            with chart_col1:

                st.markdown(
                    "##### 비율 및 점수 지표"
                )


                fig_rates = px.bar(

                    df_rates,

                    x="Metric",

                    y="Score",

                    color="Type",

                    barmode="group",

                    color_discrete_map={
                        "Standard": "#90CAF9",
                        "Personalized": "#1565C0"
                    },

                    text_auto=".2f"
                )


                fig_rates.update_layout(

                    xaxis_title="",

                    yaxis_title="Score",

                    legend_title="",

                    xaxis_tickangle=-35,

                    yaxis_range=[0, 1],

                    height=430,

                    margin=dict(
                        l=20,
                        r=20,
                        t=30,
                        b=100
                    )
                )


                st.plotly_chart(
                    fig_rates,
                    use_container_width=True
                )


            with chart_col2:

                st.markdown(
                    "##### 평균 문장 길이"
                )


                fig_length = px.bar(

                    df_length,

                    x="Metric",

                    y="Score",

                    color="Type",

                    barmode="group",

                    color_discrete_map={
                        "Standard": "#90CAF9",
                        "Personalized": "#1565C0"
                    },

                    text_auto=".2f"
                )


                fig_length.update_layout(

                    xaxis_title="",

                    yaxis_title="어절/단어 수",

                    legend_title="",

                    height=430
                )


                st.plotly_chart(
                    fig_length,
                    use_container_width=True
                )


        # ==================================================
        # TAB 2
        # ==================================================

        with tab2:

            st.markdown(
                "##### 정량 지표 비교 데이터"
            )

            st.dataframe(
                df.set_index("Metric"),
                use_container_width=True
            )


        # ==================================================
        # TAB 3
        # ==================================================

        with tab3:

            st.markdown(
                "##### 실제 탐지된 AI 클리셰"
            )


            st.markdown(
                "**Standard**"
            )

            if standard_found:

                for phrase in standard_found:

                    st.write(
                        f"- {phrase}"
                    )

            else:

                st.write(
                    "탐지된 클리셰 없음"
                )


            st.divider()


            st.markdown(
                "**Personalized**"
            )


            if personalized_found:

                for phrase in personalized_found:

                    st.write(
                        f"- {phrase}"
                    )

            else:

                st.write(
                    "탐지된 클리셰 없음"
                )


            st.divider()


            st.markdown(
                "##### 현재 사용 중인 클리셰 사전"
            )


            for category, phrases in (
                ENGLISH_AI_CLICHES.items()
            ):

                st.markdown(
                    f"**{category}**"
                )

                st.write(
                    ", ".join(phrases)
                )


# ==================================================
# Run
# ==================================================

if __name__ == "__main__":
    main()