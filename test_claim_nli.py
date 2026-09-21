from src.verify import extract_claims, nli_check


source_text = """
AI는 생산성을 향상시킬 수 있지만,
일부 산업에서는 고용 감소를 초래할 가능성도 있다.
"""

translated_text = """
AI can improve productivity, but it will reduce employment in all industries.
"""


claims = extract_claims(source_text, language="ko")


print("=== Claim + NLI Test ===\n")

for i, claim in enumerate(claims, start=1):
    result = nli_check(
        translated_text,
        claim
    )

    print(f"Claim {i}: {claim}")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print()