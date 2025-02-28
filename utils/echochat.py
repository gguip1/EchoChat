from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableSequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class EchoChat:
    def __init__(self, df_chat, speaker, model_type, history_limit=5):
        """
        EchoChat Initialization
        :param df_chat: DataFrame containing chat history
        :param speaker: Target speaker for the chatbot
        :param model_type: Model type for the chatbot ('gemini' or 'llama')
        :param history_limit: Number of previous messages to consider
        """
        self.DEBUG = False
        
        self.df_chat = df_chat
        self.speaker = speaker
        self.history_limit = history_limit
        self.messages = []
        
        self.top_words, self.top_expressions, self.avg_length = self.analyze_speaker_style()
        
        if model_type == "gemini":
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY, streaming=True)
        else:
            self.llm = Ollama(model="llama3.1:8b", streaming=True)
        
        # self.prompt = PromptTemplate(
        #     input_variables=["speaker", "avg_length", "top_words", "history_text", "recent_conversations", "user_input"],
        #     template="""
        #     You must speak like "{speaker}".
        #     "{speaker}"'s conversation style is as follows:
        #     - Average sentence length: {avg_length} characters.
        #     - Frequently used words: {top_words}

        #     🔹 Your goal is to create a meaningful and engaging conversation.
        #     - Your responses should be **at least 10 characters long** to ensure a natural conversation flow.
        #     - If the user asks a question, respond with **a clear and meaningful answer, not just a short phrase**.
        #     - Avoid repeating the same response multiple times.
        #     - Use past conversations as context to make the conversation feel natural.
        #     - If the user asks "뭐해?" or similar, do not just repeat the question. Instead, describe what you (the AI) might be doing.

        #     Past conversation history (for reference):
        #     {history_text}

        #     Recent conversation:
        #     {recent_conversations}

        #     User Input:
        #     {user_input}
        #     """
        # )
        
        self.prompt = PromptTemplate(
            input_variables=["speaker", "history_text", "recent_conversations", "user_input"],
            template="""
            You must speak like "{speaker}".
            Below is a conversation history of how "{speaker}" speaks.

            🔹 **Instructions**:
            - Maintain the tone, style, and sentence length used in the conversation history.
            - Respond **in a single line without spaces** between words.
            - Do **not** repeat sentences exactly but generate new responses in the same style.
            - Keep the responses **contextually relevant and natural**.

            🔹 **{speaker}'s Conversation History**:
            {history_text}

            🔹 **Recent Conversation**:
            {recent_conversations}

            🔹 **User Input**:
            {user_input}
            """
        )

        self.chain = self.prompt | self.llm
        
    def analyze_speaker_style(self):
        """Analyze the speaker's style based on the chat history"""
        speaker_messages = self.df_chat[self.df_chat["Speaker"] == self.speaker]["Message"].astype(str)

        if speaker_messages.empty:
            return [], [], 15

        common_expressions = ["ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅇㅇ", "ㄷㄷ", "ㄱㄱ", "ㅜㅜ"]
        expression_counts = {exp: sum(1 for msg in speaker_messages if exp in msg) for exp in common_expressions}
        top_expressions = [exp for exp, count in expression_counts.items() if count > 0]

        word_counts = Counter(" ".join(speaker_messages).split())
        top_words = [word for word, count in word_counts.most_common(50) if word not in common_expressions]

        avg_length = sum(len(msg) for msg in speaker_messages) / len(speaker_messages)

        return top_words, top_expressions, str(int(avg_length))
    
    def find_similar_message(self, user_input, top_n=3):
        """Find the most similar message from the speaker's chat history using TF-IDF"""
        speaker_messages = self.df_chat[self.df_chat["Speaker"] == self.speaker]["Message"].astype(str)
        user_messages = self.df_chat[self.df_chat["Speaker"] != self.speaker]["Message"].astype(str)

        if speaker_messages.empty or user_messages.empty:
            return []

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(user_messages)
        user_tfidf = vectorizer.transform([user_input]) 

        similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1] 

        responses = []
        for idx in top_indices:
            if idx + 1 < len(speaker_messages):
                responses.append(speaker_messages.iloc[idx + 1])

        return responses
    
    def generate_response(self, user_input):
        """Generate a response"""
        history_text = "\n".join(self.find_similar_message(user_input, top_n=5))
        recent_conversations = "\n".join([chat["content"] for chat in self.messages[-self.history_limit:]])
        
        input_data = {
            "speaker": self.speaker,
            "avg_length": self.avg_length,
            "top_words": ", ".join(self.top_words),
            "top_expressions": ", ".join(self.top_expressions),
            "history_text": history_text,
            "recent_conversations": recent_conversations,
            "user_input": user_input
        }
        
        if self.DEBUG:
            print(input_data)
        
        response = self.chain.run(input_data)

        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def generate_response_stream(self, user_input):
        """Generate a response (streaming version)"""
        history_text = "\n".join(self.find_similar_message(user_input, top_n=5))
        recent_conversations = "\n".join([chat["content"] for chat in self.messages[-self.history_limit:]])
        
        input_data = {
            "speaker": self.speaker,
            "avg_length": self.avg_length,
            "top_words": ", ".join(self.top_words),
            "top_expressions": ", ".join(self.top_expressions),
            "history_text": history_text,
            "recent_conversations": recent_conversations,
            "user_input": user_input
        }
        
        if self.DEBUG:
            print(input_data)
        
        response_stream = self.chain.stream(input_data)
        
        bot_response = ""
        for chunk in response_stream:
            text_chunk = chunk.content
            bot_response += text_chunk
            yield text_chunk
        
        self.messages.append({"role": "assistant", "content": bot_response})
