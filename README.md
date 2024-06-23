## Simple Recommenation Engine


### 데이터셋
- [Kaggle: Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)
    - anime.csv
    - anime_with_synopsis.csv
    - The above two datasets are merged by using 
    
    ```python
    merged_anime = pd.merge(anime_with_synopsis, anime, how="left").dropna()
    ```
- [Kaggle: Webtoon Comics Dataset](https://www.kaggle.com/datasets/swarnimrai/webtoon-comics-dataset)
---

### 개요
- 본 프로젝트는 사용자의 애니메이션 경험에 기반하여 사용자가 흥미로울만한 웹툰을 추천하는 엔진이다. 이 추천 시스템은 단순히 정수형태의 사용자의 ID를 받아 해당 사용자에게 웹툰을 추천한다. 애니메이션 데이터 내의 사용자가 Rating 9점 이상을 준 애니메이션 정보를 활용하여 웹툰의 정보와 코사인 유사도를 측정, 유사한 웹툰을 추천하는 서비스이다.

### Techical Stack
- 추천 모델: Hugging Face의 Sentens-Transformer에서 제공하는 임베더를 활용하며 그 중에서 간단한 `paraphrase-distilroberta-base-v1`를 활용하여 임베딩을 수행하여 유사도 기반으로 웹툰을 추천
    - OpenAI의 클로즈드 임베딩과 그 외 다양한 오픈소스 임베딩 모델을 사용할 수 있지만, 데이터의 규모가 크지 않기 때문에 경량화된 모델을 선택하고, 문장 유사도 계산이 용이한 `paraphrase-distilroberta-base-v1`를 사용.
    - 문장은 Pandas DataFrame에서 아래와 같은 방식으로 생성한다.
    ```python
    def get_webtoon_description(row):
    # Create description for one row
    description = (
        f"{row['Name']} has a subscribers {row['Subscribers']}. \n" 
        f"Summary: {row['Summary']}.\n"
        f"It was produced by {row['Writer']}.\n"
        f"Its genres are {row['Genre']}.\n"
    )
    return clean_text(description)
    ```
- API: 간단한 Serving 및 데이터 전송을 위해 빠른 API 구현에 최적화된 FASTAPI를 사용
- 사용자 인터페이스: 프로로타입 시험을 위해 간단히 프론트엔드 기능을 구현할 수 있는 Streamlit을 사용

### 해결 과제
- 문제점: 웹툰 데이터에는 아이디, 작가, 장르, 평가, 구독, 요약문 등이 포함되어 있지만, 어떤 사용자가 어떻게 평가했는지에 대한 데이터가 없어 사용자 경험을 반영한 추천 모델을 만들기가 어려움.
- 해결 방안: 애니메이션 데이터에는 사용자 경험이 포함되어 있음. 웹툰과 애니메이션의 경우 미디어는 다르지만, 웹툰이 애니메이션화가 되기도 하듯, 장르 컨텐츠의 유사성이 존재하기 때문에 특정 장르의 애니메이션을 선호하는 사용자가 비슷한 장르의 웹툰을 선호할 것이라는 가설을 세울 수 있음.
- 구현 방안: 임베딩 모델을 통해 애니메이션 데이터에 대한 Description을 학습하고, 이와 유사하게 웹툰의 데이터에 대한 Description을 학습하여, 두 개의 임베딩 데이터를 구축하여, 임베딩 데이터 간의 유사성을 계산하여 애니메이션과 웹툰과의 유사성을 측정할 수 있음.
- 결과: 애니메이션 데이터의 사용자가 애니메이션을 평가한 결과를 이용하여 사용자가 선호하는 애니메이션을 추출, 추출한 애니메이션과 유사한 웹툰을 추천하는 형태로 기능을 구현.

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant FastAPI
    participant RecommendationFunction

    User->>Streamlit: 사용자 ID 입력
    Streamlit->>FastAPI: 사용자 ID 전달
    FastAPI ->> RecommendationFunction: 사용자 ID 및 데이터 전달
    RecommendationFunction ->> RecommendationFunction: 애니메이션과 웹툰 데이터와의 코사인 유사도 계산
    RecommendationFunction ->> RecommendationFunction: 애니메이션 데이터 셋의 사용자 ID의 선호에 따라 웹툰 추천
    RecommendationFunction-->>FastAPI: 추천 웹툰 반환
    FastAPI-->>Streamlit: 추천 웹툰 전송
    Streamlit-->>User: 추천 웹툰 디스플레이
```

### Getting Started
```
Python: 3.11.9 and Install on your system requirements.txt.
```
- 설치 방법
    1. git clone https://github.com/Rakdol/Simple-Recommendation-Engine.git
    2. 프로젝트 폴더로 이동
    3. pip install -r requirements.txt

- Running FASTAPI
    - 코드상에 디폴트 포트가 5020으로 지정, 편의에 따라 수정 가능
    ```bash
    uvicorn app:app --reload --port=5020
    ```
- Running Streamlit
    - 포트는 편의에 따라 수정 가능
    ```bash
    streamlit run streamlit_board.py --server.port 8002  
    ```
- ScreenShot:
    - 단순한 정수 ID 기반의 데이터 입력
    - ID에 맞는 웹툰 추천 및 추천의 소스가 된 애니메이션 목록 표출
<p align="center">
<img src="./assets/image.png" width="500" height="500">
<img src="./assets/image-1.png" width="500" height="500">
</p>

### 한계점
- 경량화 모델 기반의 임베딩이지만 최적화 문제로 추천까지 실행시간이 다소 있음.
- 웹툰과 애니메이션 데이터의 언어가 영어로 되어 있어 한국어 명령 등 사용이 어려움.
- 애니메이션의 데이터가 웹툰에 비해 비대하고 장르 구분 등 보다 구체적인 설명이 애니메이션 데이터에는 제공되고 있지만 웹툰 데이터에 대해서는 상대적으로 적은 정보가 존재.
- 유사도가 높게 나와도 정말 유사한지에 대한 의문이 있음. 
- 현재 단순하게 사용자의 ID 만을 입력을 받도록 되어 있지만, 향 후 장르에 대한 설명 등의 자연어 처리와 VectorDB와 RAG를 통한 시맨틱 검색을 통해 추천을 구현할 수도 있을 것 같음.