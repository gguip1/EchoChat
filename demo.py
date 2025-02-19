import streamlit as st
import pandas as pd
from utils import Preprocessor
from utils import EOChat

st.set_page_config(page_title="카카오톡 챗봇")

# 🔹 세션 상태 초기화
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = Preprocessor()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_speaker" not in st.session_state:
    st.session_state.selected_speaker = None
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# 🔹 사용자 이름 입력
with st.sidebar:
    st.title("🔧 설정")
    user_name = st.text_input("당신의 이름을 입력하세요:", value=st.session_state.user_name)
    
    if user_name:
        st.session_state.user_name = user_name
    
    uploaded_file = st.file_uploader("📂 카카오톡 대화 내역을 업로드하세요", type=["txt"])

st.title("💬 카카오톡 챗봇")

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")
    lines = file_contents.split("\n")
    df = pd.DataFrame(st.session_state.preprocessor.clean_data(lines), columns=["Date", "Time", "Speaker", "Message"])
    
    # 🔹 데이터 표시
    st.subheader("📌 대화 데이터")
    st.dataframe(df, use_container_width=True)
    
    # 🔹 대화할 Speaker 선택
    speakers = df["Speaker"].unique().tolist()
    selected_speaker = st.sidebar.selectbox("💬 대화할 상대를 선택하세요:", speakers, index=0)
    
    if selected_speaker != st.session_state.selected_speaker:
        st.session_state.selected_speaker = selected_speaker
        st.session_state.messages = []
        st.session_state.chatbot = EOChat(df, selected_speaker)
    
    st.subheader(f"🤖 {selected_speaker}과 대화 중...")
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            role = "👤 " + user_name if msg["role"] == "user" else "🤖 " + selected_speaker
            align = "text-align: right;" if msg["role"] == "user" else "text-align: left;"
            bg_color = "#DCF8C6" if msg["role"] == "user" else "#EAEAEA"
            
            st.markdown(
                f"""
                <div style="{align}; padding: 10px; margin: 5px; border-radius: 10px; background-color: {bg_color}; max-width: 70%; display: inline-block;">
                    <b>{role}</b><br>{msg["content"]}
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    # 🔹 사용자 입력
    user_input = st.chat_input("메시지를 입력하세요...")
    
    if user_input:
        with st.chat_message(f"👤 {user_name}"):
            st.markdown(
                f"""
                <div style="text-align: right; padding: 10px; margin: 5px; border-radius: 10px; background-color: #DCF8C6; max-width: 70%; display: inline-block;">
                    <b>👤 {user_name}</b><br>{user_input}
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with st.chat_message(f"🤖 {selected_speaker}"):
            response_container = st.empty()
            full_response = ""
            
            for chunk in st.session_state.chatbot.generate_response_stream(user_input):
                full_response += chunk
                response_container.markdown(
                    f"""
                    <div style="text-align: left; padding: 10px; margin: 5px; border-radius: 10px; background-color: #EAEAEA; max-width: 70%; display: inline-block;">
                        <b>🤖 {selected_speaker}</b><br>{full_response}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            st.session_state.messages.append({"role": st.session_state.user_name, "content": user_input})
            st.session_state.messages.append({"role": selected_speaker, "content": full_response})