import json
import numpy as np
from google import genai
import torch
from .embeddings import embed_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification


client = genai.Client()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)

    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )


def semantic_similarity(source_text: str, translated_text: str) -> float:
    source_vector = embed_text(
        source_text,
        task_type="RETRIEVAL_DOCUMENT"
    )

    translated_vector = embed_text(
        translated_text,
        task_type="RETRIEVAL_DOCUMENT"
    )

    return cosine_similarity(source_vector, translated_vector)


def extract_claims(text: str, language: str = "ko") -> list[str]:
    prompt = f"""
You are extracting atomic claims from text.

Rules:
1. Extract only claims explicitly supported by the input.
2. Do not add new information.
3. Preserve negation, numbers, modality, conditions, and comparisons.
4. Each claim should contain only one independently verifiable proposition.
5. Return only JSON in this format:
{{"claims": ["claim 1", "claim 2"]}}

Language of the input: {language}

Text:
{text}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    raw_text = response.text.strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]

    if raw_text.startswith("```"):
        raw_text = raw_text[3:]

    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]

    raw_text = raw_text.strip()

    result = json.loads(raw_text)

    return result["claims"]

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)


def nli_check(premise: str, hypothesis: str) -> dict:
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = nli_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    labels = nli_model.config.id2label

    scores = {
        labels[i]: probabilities[i].item()
        for i in range(len(probabilities))
    }

    best_label = max(scores, key=scores.get)

    return {
        "label": best_label,
        "confidence": scores[best_label],
        "scores": scores
    }
    
def verify_claims(source_text: str, translated_text: str) -> dict:
    source_claims = extract_claims(source_text, language="ko")
    translated_claims = extract_claims(translated_text, language="en")

    source_results = []
    translated_results = []

    # Direction 1
    # 번역문이 원문의 Claim을 지원하는가?
    # -> Claim Coverage / Omission 검사
    for claim in source_claims:
        result = nli_check(
            translated_text,
            claim
        )

        source_results.append({
            "claim": claim,
            "label": result["label"],
            "confidence": result["confidence"]
        })

    # Direction 2
    # 원문이 번역문의 Claim을 지원하는가?
    # -> Unsupported Addition / 과장 검사
    for claim in translated_claims:
        result = nli_check(
            source_text,
            claim
        )

        translated_results.append({
            "claim": claim,
            "label": result["label"],
            "confidence": result["confidence"]
        })

    return {
        "source_claims": source_results,
        "translation_claims": translated_results
    }

def verify_translation(source_text: str, translated_text: str) -> dict:
    similarity = semantic_similarity(
        source_text,
        translated_text
    )

    claim_results = verify_claims(
        source_text,
        translated_text
    )

    return {
        "semantic_similarity": similarity,
        "source_claims": claim_results["source_claims"],
        "translation_claims": claim_results["translation_claims"]
    }