import re
import pandas as pd

class Preprocessor:
    def __init__(self, remove_keywords=None):
        self.remove_keywords = remove_keywords or [
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
        
        self.date_pattern = re.compile(r"^-+\s(\d{4}년 \d{1,2}월 \d{1,2}일) .+ -+$")
        self.message_pattern = re.compile(r"^\[(.+?)\] \[(오전|오후) (\d{1,2}:\d{2})\] (.+)$")

    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = file.readlines()
        return raw_data

    def clean_data(self, raw_data):
        chat_data = []
        current_date = None
        
        for line in raw_data:
            line = line.strip()
            
            date_match = self.date_pattern.match(line)
            if date_match:
                current_date = date_match.group(1).replace("년 ", "-").replace("월 ", "-").replace("일", "")
                continue
            
            message_match = self.message_pattern.match(line)
            if message_match and current_date:
                speaker, period, time, message = message_match.groups()
                
                hour, minute = map(int, time.split(":"))
                if period == "오후" and hour != 12:
                    hour += 12
                elif period == "오전" and hour == 12:
                    hour = 0
                
                formatted_time = f"{hour:02d}:{minute:02d}"
                
                if any(keyword in message for keyword in self.remove_keywords):
                    continue
                else:
                    chat_data.append([current_date, formatted_time, speaker, message])
        
        return chat_data
    