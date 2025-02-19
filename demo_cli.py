import pandas as pd
import argparse
import time

from utils import EOChat
from utils import Preprocessor

def main(file_path):
    raw_data = Preprocessor().load_data(file_path)
    df = pd.DataFrame(Preprocessor().clean_data(raw_data=raw_data), columns=["Date", "Time", "Speaker", "Message"])

    speakers = df["Speaker"].unique().tolist()

    for index, value in enumerate(speakers):
        print(f"{index}. {value}")
    
    print("💬 대화할 상대의 번호 입력 :", end=" ")
    try:
        selected_speaker = speakers[int(input())]
    except ValueError:
        print("잘못된 입력입니다. 대화할 상대의 번호를 입력하세요.")
    
    history_limit = int(input("📈 대화 기록 크기 설정 (숫자 입력) : "))
    
    chatbot = EOChat(df, selected_speaker, model="llama3.1:8b" ,history_limit=history_limit)

    while True:
        user_input = input("사용자: ")
        print(selected_speaker, end=": ", flush=True)
        for chunk in chatbot.generate_response_stream(user_input):
            print(chunk, end="", flush=True)
            time.sleep(0.05)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    main(args.file_path)
