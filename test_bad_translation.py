from src.verify import verify_translation


korean_text = """
AI는 생산성을 향상시킬 수 있지만, 일부 산업에서는 고용 감소를 초래할 가능성도 있다.
따라서 AI의 도입이 항상 긍정적인 결과만 가져온다고 단정하기는 어렵다.
"""


# 일부러 의미를 왜곡한 번역
bad_translation = """
Artificial intelligence always improves productivity and will increase
employment in every industry. Therefore, the implementation of artificial
intelligence produces only positive outcomes.
"""


verification = verify_translation(
    source_text=korean_text,
    translated_text=bad_translation
)


print("\n=== BAD Translation ===")
print(bad_translation)

print("\n=== Verification Result ===")

print("\nSemantic Similarity:")
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
    
print("\nCritical Errors:")
for error in verification["critical_errors"]:
    print(error)

print("\nVerdict:")
print(verification["verdict"])