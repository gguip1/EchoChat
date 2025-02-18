import pandas as pd
import re

remove_kewords = [
    "이모티콘", 
    "사진", 
    "동영상", 
    "삭제된 메시지입니다",
    "송금",
    "파일:",
    "(안내)",
    "받았어요.",
    "시작합니다!",
    "보냈어요."
    ]

def load_data(file_path):
    """파일을 읽어서 라인별로 리스트로 반환합니다."""
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = file.readlines()
    return raw_data

def clean_kakao_data(raw_data):
    """카카오톡 대화 내용을 정제합니다."""
    chat_data = []
    current_date = None
    
    date_pattern = re.compile(r"^-+\s(\d{4}년 \d{1,2}월 \d{1,2}일) .+ -+$")
    message_pattern = re.compile(r"^\[(.+?)\] \[(오전|오후) (\d{1,2}:\d{2})\] (.+)$")

    for line in raw_data:
        line = line.strip()
        
        date_match = date_pattern.match(line)
        if date_match:
            current_date = date_match.group(1).replace("년 ", "-").replace("월 ", "-").replace("일", "")
            continue
        
        message_match = message_pattern.match(line)
        if message_match and current_date:
            speaker, period, time, message = message_match.groups()
            
            hour, minute = map(int, time.split(":"))
            if period == "오후" and hour != 12:
                hour += 12
            elif period == "오전" and hour == 12:
                hour = 0
            
            formatted_time = f"{hour:02d}:{minute:02d}"
            
            if any(keyword in message for keyword in remove_kewords):
                continue
            else:
                chat_data.append([current_date, formatted_time, speaker, message])
    
    return chat_data

def save_data(chat_data, save_path):
    """정제된 대화 데이터를 CSV 파일로 저장합니다."""
    df = pd.DataFrame(chat_data, columns=["Date", "Time", "Speaker", "Message"])
    df.to_csv(save_path, index=False)


save_data(clean_kakao_data(load_data("data_2.txt")), "cleaned_data_2.csv") 