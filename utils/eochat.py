import pandas as pd
import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter

class EOChat:
    def __init__(self, df_chat, speaker, model="llama3.1:8b", history_limit=5):
        """
        EchoChat Initialization
        :param model: Ollama model name
        :param history_limit: Number of previous messages to consider
        """
        self.df_chat = df_chat
        self.speaker = speaker
        self.model = model
        self.history_limit = history_limit
        self.messages = []
        
        self.top_words, self.top_expressions, self.avg_length = self.analyze_speaker_style()
    
    def add_message(self, role, content):
        """Add a message to the chat history"""
        self.messages.append({"role": role, "content": content})
        
        if len(self.messages) > self.history_limit:
            self.messages = self.messages[-self.history_limit:]
    
    def find_similar_message(self, user_input, top_n=3):
        """Find the most similar message from the speaker's chat history using TF-IDF"""
        speaker_messages = self.df_chat[self.df_chat["Speaker"] == self.speaker]["Message"].astype(str)
        
        if speaker_messages.empty:
            return []
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(speaker_messages)
        
        user_tfidf = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        return speaker_messages.iloc[top_indices].tolist()
    
    def analyze_speaker_style(self):
        """Analyze the speaker's style based on the chat history"""
        speaker_messages = self.df_chat[self.df_chat["Speaker"] == self.speaker]["Message"].astype(str)
        
        if speaker_messages.empty:
            return [], [], 15
        
        word_counts = Counter(" ".join(speaker_messages).split())
        top_words = [word for word, count in word_counts.most_common(10)]
        
        common_expressions = ["ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅇㅇ", "ㄷㄷ", "ㄱㄱ", "ㅜㅜ"]
        expression_counts = {exp: sum(1 for msg in speaker_messages if exp in msg) for exp in common_expressions}
        top_expressions = [exp for exp, count in expression_counts.items() if count > 0]

        avg_length = sum(len(msg) for msg in speaker_messages) / len(speaker_messages)
        
        return top_words, top_expressions, str(int(avg_length))
    
    def generate_prompt(self, user_input):
        """Generate a prompt for the Ollama model"""
        similar_chats = self.find_similar_message(user_input, top_n=5)
        history_text = "\n".join(similar_chats)
        
        # prompt = f"""
        # 너는 "{self.speaker}"처럼 말해야 해.
        # "{self.speaker}"의 대화 스타일은 다음과 같아:
        # - 평균적인 문장 길이: {self.avg_length}자 (너도 비슷한 길이로 답변해)
        # - 자주 쓰는 단어: {", ".join(self.top_words)}
        # - 자주 쓰는 표현: {", ".join(self.top_expressions)}
        
        # 사용자와 자연스럽게 대화를 이어나가도록 답변을 생성해줘.
        # - 같은 문장을 반복하지 말고, 상황에 따라 적절하게 변형해서 응답해줘.
        # - 단순한 질문-응답 형태가 아니라, 자연스러운 대화로 이어지도록 응답을 생성해줘.
        # - 사용자의 질문에 대한 직접적인 대답을 하되, 이전 대화와의 연결성을 고려해.
        # - 새로운 방식으로 답변을 생성하되, "{self.speaker}"의 말투를 유지해야 해.
        
        
        # 과거 대화 기록 (참고용):
        # {history_text}
        
        # 최근 대화:
        # {self.get_recent_conversations()}
        
        # """
        
        prompt = f"""
        
        You must speak like "{self.speaker}".
        "{self.speaker}"'s conversation style is as follows:
        - Average sentence length: {self.avg_length} characters (please respond with similar length)
        - Frequently used words: {", ".join(self.top_words)}
        - Common expressions: {", ".join(self.top_expressions)}

        Generate responses to engage in a natural conversation with the user:
        - Do not repeat the same sentence; adapt responses according to the context.
        - Aim for a flow in conversation rather than simple question-answer exchanges.
        - Provide direct answers to the user's questions while considering the continuity from previous dialogue.
        - Create new responses but maintain "{self.speaker}"'s tone and style.

        Past conversation history (for reference):
        {history_text}

        Recent conversation:
        {self.get_recent_conversations()}
        
        """
        
        return prompt
    
    def generate_response(self, user_input):
        """Generate a response using the Ollama model"""
        self.add_message("user", user_input)
        
        prompt = self.generate_prompt(user_input)
        
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 대화 스타일을 유지하면서 자연스럽게 응답하는 AI야."},
                {"role": "user", "content": prompt}
            ]
        )
        
        bot_response = response["message"]["content"]
        self.add_message(self.speaker, bot_response)
        
        return response
    
    def generate_response_stream(self, user_input):
        """Generate a response using the Ollama model (streaming version)"""
        self.add_message("user", user_input)
        
        prompt = self.generate_prompt(user_input)
        
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 대화 스타일을 유지하면서 자연스럽게 응답하는 AI야."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        bot_response = ""
        for chunk in response:
            content = chunk["message"]["content"]
            bot_response += content
            yield content
    
    def get_recent_conversations(self):
        """Get the recent conversations in the chat history"""
        return "\n".join([f"{chat['role']}: {chat['content']}" for chat in self.messages[-self.history_limit:]])
    
    def get_chat_history(self):
        """Get the chat history"""
        return self.messages
    
    def get_info(self):
        """Get the information about the speaker"""
        return {
            "Speaker": self.speaker,
            "Top Words": self.top_words,
            "Top Expressions": self.top_expressions,
            "Average Length": self.avg_length
        }