"""영어 AI 클리셰 사전 (팀 공용). app.py와 evaluate_metrics.py가 같은 사전을 쓰도록 한 곳에 둡니다.

nltk 같은 추가 패키지 없이 import할 수 있게 분리했습니다.
"""

ENGLISH_AI_CLICHES = {
    # 1. 인위적 담화 표지 및 전환구 (Discourse Markers & Transitions)
    "transitions": [
        "furthermore", "moreover", "in addition", "additionally", "notably",
        "consequently", "subsequently", "it is worth noting that",
        "it is noteworthy that", "importantly", "in summary", "to summarize",
        "in conclusion", "all in all", "ultimately", "moving forward"
    ],
    
    # 2. 과도한 강조 및 전형적 수식어 (Hallmark Adjectives & Adverbs)
    "elevated_modifiers": [
        "pivotal", "crucial", "vital", "paramount", "essential", "indispensable",
        "multifaceted", "nuanced", "intricate", "meticulous", "comprehensive",
        "robust", "dynamic", "vibrant", "seamless", "transformative", "groundbreaking",
        "profound", "imperative", "ubiquitous", "compelling"
    ],
    
    # 3. AI 특유의 빈출 동사 및 관용적 동사구 (Frequent Verbs & Verb Phrases)
    "cliche_verbs_and_phrases": [
        "delve into", "delves into", "delving into",
        "plays a crucial role", "play a crucial role",
        "plays a pivotal role", "play a pivotal role",
        "foster", "fosters", "fostering",
        "underscore", "underscores", "underscoring",
        "highlight", "highlights", "highlighting",
        "shed light on", "sheds light on",
        "bridge the gap", "bridges the gap",
        "navigate", "navigates", "navigating",
        "harness", "harnesses", "harnessing",
        "unlock", "unlocks", "unlocking",
        "spearhead", "spearheads", "spearheading",
        "epitomize", "epitomizes", "epitomizing"
    ],
    
    # 4. 상투적 은유 및 명사구 (Metaphorical Clichés & Tropes)
    "metaphors_and_tropes": [
        "testament to", "a testament to",
        "tapestry", "rich tapestry", "complex tapestry",
        "beacon", "beacon of",
        "cornerstone", "cornerstone of",
        "double-edged sword", "a double-edged sword",
        "game changer", "a game changer",
        "paradigm shift",
        "ever-evolving", "ever-changing",
        "in the realm of", "in today's world",
        "in this digital age", "fast-paced world"
    ]
}
