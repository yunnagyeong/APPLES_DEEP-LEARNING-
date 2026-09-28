from src.store import read_examples
from src import style_features as sf
from src.pipeline import translate_with_verification

examples = read_examples()
texts = [ex["text"] for ex in examples]
profile = sf.extract_style_profile(texts)

korean_input = "내 생각엔 이 설계가 더 낫다."
result = translate_with_verification(korean_input, profile, max_retries=1)

print(f"verdict: {result['verdict']}, attempts: {result['attempts']}")
print(f"최종 번역: {result['translation']}")
print()
for i, v in enumerate(result["verification_history"], start=1):
    print(f"--- 시도 {i}: {v['verdict']} ---")
    for r in v["failure_reasons"]:
        print(f"  [{r['type']}] {r['claim']}")
