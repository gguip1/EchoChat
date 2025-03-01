import pandas as pd
import argparse
import time

from utils import EchoChat
from utils import Preprocessor

def main(file_path):
    raw_data = Preprocessor().load_data(file_path)
    df = pd.DataFrame(Preprocessor().clean_data(raw_data=raw_data), columns=["Date", "Time", "Speaker", "Message"])

    speakers = df["Speaker"].unique().tolist()

    for index, value in enumerate(speakers):
        print(f"{index + 1}. {value}")
    
    print("💬 대화할 상대의 번호 입력 :", end=" ")
    try:
        selected_speaker = speakers[int(input()) - 1]
    except ValueError:
        print("잘못된 입력입니다. 대화할 상대의 번호를 입력하세요.")
    
    print("💬 사용할 모델을 선택하세요.")
    print("1. gemini-2.0-flash-lite")
    print("2. llama3.1:8b")
    print("💬 모델 번호 입력 :", end=" ")
    model_type = int(input())
    
    if model_type == 1:
        model_type = "gemini"
    else:
        model_type = "llama"
    
    print("💬 Debug 모드를 사용하시겠습니까? (y/n) :", end=" ")
    debug = input()
    if debug == "y":
        debug = True
    else:
        debug = False

    chatbot = EchoChat(df, selected_speaker, model_type=model_type, debug=debug)

    while True:
        user_input = input("사용자: ")
        print(selected_speaker, end=": ", flush=True)
        for chunk in chatbot.generate_response_stream(user_input):
            print(chunk, end="", flush=True)
            time.sleep(0.1)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()

    main(args.file_path)
