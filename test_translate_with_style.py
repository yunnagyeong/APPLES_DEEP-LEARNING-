from src.generate import translate_with_style
from src.verify import verify_translation


korean_text = """
AI는 생산성을 향상시킬 수 있지만, 일부 산업에서는 고용 감소를 초래할 가능성도 있다.
따라서 AI의 도입이 항상 긍정적인 결과만 가져온다고 단정하기는 어렵다.
"""

style_profile = {
    "sentence_length_avg": 8.0,
    "formality_score": 0.8,
    "directness_score": 0.6,
    "logical_connective_rate": 0.7,
    "hedging_rate": 0.5,
    "conclusion_position": "back",
    "example_usage_rate": 0.2,
    "lexical_diversity": 0.85,
    "paragraph_length_avg": 3.0,
    "source_example_count": 20,
}


# 1. Generation
translated_text = translate_with_style(
    korean_text=korean_text,
    style_profile=style_profile
)

print("\n=== Personalized English Translation ===")
print(translated_text)


# 2. Verification
verification = verify_translation(
    source_text=korean_text,
    translated_text=translated_text
)

print("\n=== Verification Result ===")
print("Semantic Similarity:")
print(verification["semantic_similarity"])

print("\nSource Claims:")
for claim in verification["source_claims"]:
    print(claim)

print("\nTranslation Claims:")
for claim in verification["translation_claims"]:
    print(claim)
    
print("\nFailure Reasons:")
for reason in verification["failure_reasons"]:
    print(reason)