import joblib
from sentence_transformers import SentenceTransformer

# 모델 불러오기
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 샘플 문장 리스트
sentences = [
    "안녕하세요, 오늘 날씨가 정말 좋네요.",
    "여기서 가장 가까운 커피숍은 어디인가요?",
    "자연어 처리는 흥미로운 분야입니다."
]

# 임베딩 추출
embeddings = model.encode(sentences)

# 임베딩 결과 출력
for i, sentence in enumerate(sentences):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embeddings[i]}")
    print("\n")

# 모델을 .pkl 파일로 저장
joblib.dump(model, './sbert_model.pkl')