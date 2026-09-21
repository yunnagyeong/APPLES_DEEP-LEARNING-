
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

# --- 사전 정의 (cliche_lexicon.py가 없을 경우를 대비한 내장 예시) ---
# 별도 파일로 관리할 경우 import cliche_lexicon 하시면 됩니다.
ENGLISH_AI_CLICHES = {
    "transitions": [
        "furthermore", "moreover", "in addition", "additionally", 
        "it is worth noting that", "it is noteworthy that", "in summary"
    ],
    "hedging_and_boosters": [
        "plays a crucial role", "vital to consider", "delve into", 
        "testament to", "foster", "pivotal", "beacon"
    ],
    "generic_closers": [
        "in conclusion", "all in all", "ultimately", "moving forward"
    ]
}

# 최초 1회 punkt 토크나이저 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass


def tokenize_english(text: str) -> list[str]:
    """영문 소문자화 및 순수 단어 토큰만 추출"""
    tokens = word_tokenize(text.lower())
    # 특수문자/구두점 제외하고 알파벳 단어만 필터링
    return [token for token in tokens if token.isalpha()]


def calculate_ttr(tokens: list[str]) -> float:
    """Type-Token Ratio (고유 단어 수 / 전체 단어 수)"""
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def calculate_mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity (MTLD)
    텍스트 길이에 왜곡되지 않는 표준 어휘 다양성 지수
    """
    if len(tokens) < 10:
        return 0.0

    def _compute_factor(token_list):
        factors = 0.0
        current_types = set()
        token_count = 0

        for token in token_list:
            token_count += 1
            current_types.add(token)
            current_ttr = len(current_types) / token_count

            if current_ttr <= ttr_threshold:
                factors += 1.0
                current_types = set()
                token_count = 0

        # 남은 단어 세그먼트에 대한 가중치 보정 (McCarthy & Jarvis 알고리즘 표준)
        if token_count > 0:
            current_ttr = len(current_types) / token_count
            if current_ttr < 1.0:
                excess_factor = (1.0 - current_ttr) / (1.0 - ttr_threshold)
                factors += excess_factor
            else:
                factors += 0.0

        return len(token_list) / factors if factors > 0 else float(len(token_list))

    forward_mtld = _compute_factor(tokens)
    backward_mtld = _compute_factor(tokens[::-1])
    return round((forward_mtld + backward_mtld) / 2.0, 2)


def calculate_cliche_density(text: str, tokens: list[str], cliche_dict: dict) -> dict:
    """영문 텍스트 내 AI 클리셰 출현 횟수 및 비율 계산"""
    lower_text = text.lower()
    total_words = len(tokens)

    flat_cliches = [phrase.lower() for category in cliche_dict.values() for phrase in category]
    detected = []

    for phrase in flat_cliches:
        # 단어 경계(\b)를 포함해 부분 일치(예: 'role' in 'controller') 방지
        pattern = rf'\b{re.escape(phrase)}\b'
        matches = re.findall(pattern, lower_text)
        if matches:
            detected.extend([phrase] * len(matches))

    cliche_count = len(detected)
    density = (cliche_count / total_words * 100) if total_words > 0 else 0.0

    return {
        "cliche_count": cliche_count,
        "cliche_density_pct": round(density, 2),
        "detected_phrases": Counter(detected)
    }


def evaluate_sample(text: str, cliche_dict: dict = ENGLISH_AI_CLICHES) -> dict:
    """단일 번역 텍스트 종합 평가"""
    tokens = tokenize_english(text)
    cliche_res = calculate_cliche_density(text, tokens, cliche_dict)

    return {
        "word_count": len(tokens),
        "ttr": calculate_ttr(tokens),
        "mtld": calculate_mtld(tokens),
        "cliche_count": cliche_res["cliche_count"],
        "cliche_density_pct": cliche_res["cliche_density_pct"],
        "detected_cliches": dict(cliche_res["detected_phrases"])
    }


if __name__ == "__main__":
    # Condition A/B 스타일 예시 (전형적인 AI 말투)
    sample_a = """
    Furthermore, it is noteworthy that this agreement plays a crucial role in regulating 
    the obligations. In addition, it is vital to consider that termination may occur.
    """

    # Condition C 스타일 예시 (구어적/개인화된 문체)
    sample_c = """
    If you don't pay on time, they can basically cancel your contract right away. 
    Just make sure you don't miss the deadline.
    """

    print("=== Condition A/B (AI-like) ===")
    res_a = evaluate_sample(sample_a)
    for k, v in res_a.items():
        print(f"  {k}: {v}")

    print("\n=== Condition C (Personalized) ===")
    res_c = evaluate_sample(sample_c)
    for k, v in res_c.items():
        print(f"  {k}: {v}")