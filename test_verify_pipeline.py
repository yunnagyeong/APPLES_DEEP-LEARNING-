from src.verify import verify_translation


source_text = """
AI는 생산성을 향상시킬 수 있지만,
일부 산업에서는 고용 감소를 초래할 가능성도 있다.
"""

translated_text = """
AI can improve productivity, but it will reduce employment in all industries.
"""


result = verify_translation(
    source_text,
    translated_text
)


print("=== Verification Pipeline Result ===")

print("\nSemantic Similarity:")
print(result["semantic_similarity"])


print("\nSource Claims:")
for item in result["source_claims"]:
    print(item)


print("\nTranslation Claims:")
for item in result["translation_claims"]:
    print(item)