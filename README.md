# EchoChat

EOChat는 카카오톡 대화 로그를 기반으로 특정 화자의 대화 스타일을 분석하고, 해당 화체를 모방하여 자연스러운 대화를 생성하는 챗봇 프로젝트입니다.

## 특징
- **대화 로그 전처리**: 날짜 및 메시지 포맷을 파싱하여 구조화된 데이터로 변환합니다.
- **스타일 분석**: 자주 사용하는 단어, 표현, 평균 문장 길이를 분석합니다.
- **대화 생성**: 분석된 스타일을 유지하면서 사용자의 입력에 맞춰 답변을 생성합니다.
- **데모 제공**: CLI와 Streamlit 기반 데모를 제공합니다.

## 사용 방법

### 사전 준비
1. [Ollama](https://ollama.ai/)를 설치합니다.
2. 터미널에서 Llama 모델을 다운로드합니다:
   ```bash
   ollama pull llama3.1:8b
   ```

### CLI 데모
1. 텍스트 파일로 카카오톡 대화 내역을 준비합니다.
2. 터미널에서 아래 명령어를 실행합니다:
   ```
   python demo_cli.py --file_path /path/to/chat.txt
   ```
3. 출력되는 목록에서 대화할 화자의 번호와 대화 기록 길이를 입력합니다.
4. 대화가 시작됩니다.

### 웹 데모 (Streamlit)
1. 웹 브라우저에서 데모를 실행하려면 아래 명령어를 실행하세요:
   ```
   streamlit run demo.py
   ```
2. 사이드바에서 사용자 이름을 입력하고, 대화 내역 파일을 업로드한 후 대화할 화자를 선택합니다.
3. 챗봇과의 대화를 시작합니다.

> **참고**: 이 프로젝트는 llama3.1:8b 모델을 사용하여 개발 및 테스트되었습니다. 다른 Ollama 모델을 사용할 경우 성능이 달라질 수 있습니다.

## 설치
필요한 패키지는 [requirements.txt](requirements.txt)를 참고하여 설치합니다:
```
pip install -r requirements.txt
```

## 프로젝트 구조
- `/utils`: 데이터 전처리 및 챗봇 관련 유틸리티 모듈
- `/demo_cli.py`: CLI 기반 데모 실행 파일
- `/demo.py`: Streamlit을 이용한 웹 데모 실행 파일


