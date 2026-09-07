# 1주차 인터페이스 약속용 가짜 데이터

# 수영 -> 강희/나경 전달 데이터 규격
DUMMY_GENERATION_RESULT = {
    "original_text": "Hello, how are you?",
    "translated_text": "안녕하세요, 잘 지내시나요?",
    "style_profile": {
        "formality": "polite",
        "tone": "soft"
    }
}

# 강희 -> 나경 전달 데이터 규격
DUMMY_VERIFICATION_RESULT = {
    "is_passed": True,
    "similarity_score": 0.95,
    "feedback": "의미 보존 양호"
}
