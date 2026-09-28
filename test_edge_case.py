from src.verify import verify_translation


#source_text = """
#AI는 일부 산업에서 고용 감소를 초래할 가능성이 있다.
#"""

#가능성 to 확정
#translated_text = """ 
#Artificial intelligence will reduce employment in some industries.
#"""

# 범위변화
#translated_text = """
#Artificial intelligence may reduce employment in all industries.
#"""

#조건절이 번역에서 누락
source_text = """
정부의 지원이 충분하다면 중소기업은 새로운 기술을 도입할 수 있다.
"""

translated_text = """
Small and medium-sized enterprises can adopt new technologies.
"""
#숫자 변화
source_text1 = """
이 정책으로 탄소 배출량이 2030년까지 30% 감소할 것으로 예상된다.
"""

translated_text1 = """
The policy is expected to reduce carbon emissions by 20% by 2030.
"""
#핵심정보 완전 누락
source_text2 = """
회사는 지난해 매출이 20% 증가했지만 영업이익은 10% 감소했다.
"""

translated_text2 = """
The company's revenue increased by 20% last year.
"""

verification = verify_translation(
    source_text=source_text2,
    translated_text=translated_text2
)


print("\n=== EDGE CASE: Modality Change ===")

print("\nSource:")
print(source_text2)

print("\nTranslation:")
print(translated_text2)

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