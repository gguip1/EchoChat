import pandas as pd
import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter

"""
프로세스 설명
1. 사용자로부터 대화 데이터 CSV 파일 경로를 입력받습니다.
2. 대화 데이터를 로드하고, 가능한 Speaker 목록을 출력합니다.
3. 사용자가 대화하고 싶은 Speaker를 선택하면, 해당 Speaker의 대화 스타일을 분석합니다.
4. 사용자가 입력한 문장과 가장 유사한 과거 대화를 찾아 출력합니다.
5. Ollama 모델을 사용하여, 선택한 Speaker의 대화 스타일에 맞는 응답을 생성합니다.
6. 생성된 응답을 출력하고, 대화를 이어나갑니다.
"""

def load_chat_data(csv_file_path):
    """CSV 파일을 로드하고 가능한 Speaker 목록을 반환"""
    df_chat = pd.read_csv(csv_file_path)
    
    if "Speaker" not in df_chat.columns or "Message" not in df_chat.columns:
        raise ValueError("CSV 파일에 'Speaker' 및 'Message' 컬럼이 필요합니다.")
    
    speakers = df_chat["Speaker"].unique().tolist()
    return df_chat, speakers

def choose_speaker(speakers):
    """사용자가 대화에서 특정 Speaker를 선택할 수 있도록 함"""
    print("📢 가능한 대화 상대 목록:")
    for i, speaker in enumerate(speakers):
        print(f"{i + 1}. {speaker}")
    
    choice = int(input("💡 대화할 대상을 선택하세요 (숫자 입력): ")) - 1
    return speakers[choice]

def analyze_speaker_style(df_chat, speaker):
    """특정 Speaker의 대화 패턴을 분석하여, 자주 쓰는 단어 및 말투 스타일을 추출"""
    speaker_messages = df_chat[df_chat["Speaker"] == speaker]["Message"].astype(str)
    
    # 단어 빈도수 계산
    word_counts = Counter(" ".join(speaker_messages).split())
    top_words = [word for word, count in word_counts.most_common(10)]
    
    # 감탄사 및 단축어 패턴 분석
    common_expressions = ["ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅇㅇ", "ㄷㄷ", "ㄱㄱ", "ㅜㅜ"]
    expression_counts = {exp: sum(1 for msg in speaker_messages if exp in msg) for exp in common_expressions}
    top_expressions = [exp for exp, count in expression_counts.items() if count > 0]
    
    # 문장 길이 분석
    avg_length = sum(len(msg) for msg in speaker_messages) / len(speaker_messages)
    style_description = f"{avg_length}" if avg_length < 10 else f"{avg_length}"
    
    return top_words, top_expressions, style_description

def find_similar_message(df_chat, target_speaker, user_input, top_n=3):
    """TF-IDF를 사용하여 특정 Speaker의 과거 대화 중 가장 유사한 문장을 검색"""
    
    # 선택한 Speaker의 대화만 사용
    speaker_messages = df_chat[df_chat["Speaker"] == target_speaker]["Message"].astype(str)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(speaker_messages)

    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # 유사도가 높은 문장 상위 N개 추출
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    return speaker_messages.iloc[top_indices].tolist()

def generate_response(df_chat, target_speaker, user_input, conversation_history):
    """Ollama 모델을 사용하여 특정 Speaker의 대화 스타일에 맞는 응답 생성"""
    
    similar_chats = find_similar_message(df_chat, target_speaker, user_input, top_n=5)
    
    # 특정 Speaker의 말투 분석
    top_words, top_expressions, style_description = analyze_speaker_style(df_chat, target_speaker)
    
    recent_conversations = "\n".join(
        [f"{chat['speaker']}: {chat['message']}" for chat in conversation_history[-5:]]
    )
    history_text = " ".join(similar_chats)
    
    prompt = f"""
    너는 "{target_speaker}"처럼 말해야 해.
    "{target_speaker}"의 대화 스타일은 다음과 같아:
    - 평균적인 문장 길이: {style_description}자 (너도 비슷한 길이로 답변해)
    - 자주 쓰는 단어: {", ".join(top_words)}
    - 자주 쓰는 표현: {", ".join(top_expressions)}

    과거 대화 기록 (참고용):
    {history_text}

    최근 대화:
    {recent_conversations}

    사용자와 자연스럽게 대화를 이어나가도록 답변을 생성해줘.
    - 같은 문장을 반복하지 말고, 상황에 따라 적절하게 변형해서 응답해줘.
    - 단순한 질문-응답 형태가 아니라, 자연스러운 대화로 이어지도록 응답을 생성해줘.
    - 사용자의 질문에 대한 직접적인 대답을 하되, 이전 대화와의 연결성을 고려해.
    - 새로운 방식으로 답변을 생성하되, "{target_speaker}"의 말투를 유지해야 해.
    
    사용자: "{user_input}"
    {target_speaker}:
    """
    
    response = ollama.chat(
        model="llama3.1:8b", 
        messages=[
                {"role": "system", "content": "너는 대화 스타일을 유지하면서 자연스럽게 응답하는 AI야."},
                {"role": "user", "content": prompt}
            ]
        )

    return response["message"]["content"]

while True:
    csv_file_path = input("💡 대화 데이터 CSV 파일 경로를 입력하세요: ")
    df_chat, speakers = load_chat_data(csv_file_path)
    
    target_speaker = choose_speaker(speakers)
    conversation_history = []
    
    while True:
        user_input = input("사용자: ")
        response = generate_response(df_chat, target_speaker, user_input, conversation_history)
        
        print(f"{target_speaker}: {response}")
        conversation_history.append({"speaker": "사용자", "message": user_input})
        conversation_history.append({"speaker": target_speaker, "message": response})