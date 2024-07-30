import os
import sys
from typing import Any
from pathlib import Path

PAKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PAKAGE_ROOT))

import requests
import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from sentence_transformers import SentenceTransformer

from src.config import DataIngestionConfig
from src.utils.utils import get_preferred_anime_from_user, get_recommenations
from src.app.schemas import UserIn, UserOut, AnimeIn, AnimeOut, WebtoonIn, WebtoonOut, QueryRecIn, QueryRecOut
from src.utils.logger import logging

def get_data(config:DataIngestionConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    webtoon_data = pd.read_csv(config.webtoon_data_path)
    rating_data = pd.read_csv(config.rating_data_path)
    anime_data = pd.read_csv(config.anime_data_path)
    
    return webtoon_data, rating_data, anime_data

def get_embeddings(config:DataIngestionConfig) -> tuple[np.array, np.array]:
    anime_embeddings = np.load(config.anime_embedding_path)
    webtoon_embeddings = np.load(config.webtoon_embedding_path)
    
    return anime_embeddings, webtoon_embeddings

def get_response(user_id:int) -> dict:
    preferred_animes = get_preferred_anime_from_user(user_id, rating)[:2] # 2 is used to reduce recommenation space
    top_itmes = get_recommenations(webtoon, anime, preferred_animes, anime_embeddings, webtoon_embeddings)
    return top_itmes

def embed_query(query_text) -> np.array:
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    return query_embedding.cpu().numpy()

def get_response_with_query(user_id:int, query:str)->dict:
    query_embedding = embed_query(query)

    # RAG 기반 검색 수행
    k = 5  # 상위 5개의 유사한 항목을 검색
    webtoon_distances, webtoon_indices = webtoon_index.search(np.array([query_embedding]), k)
    animation_distances, animation_indices = animation_index.search(np.array([query_embedding]), k)

    # 기존 사용자 ID 기반 추천
    top_items = get_response(user_id)

    # 검색 결과와 유사도 기반 추천 결과를 결합
    final_webtoon = list(set(webtoon_indices[0].tolist() + top_items["webtoon_key"]))
    final_anime = list(set(animation_indices[0].tolist() + top_items["anime_key"]))
    
    
    return {"anime_key": final_anime, "webtoon_key": final_webtoon}

def get_response_only_query(query:str) -> dict:
    # 질의 임베딩
    query_embedding = embed_query(query)

    # RAG Search
    k = 5  # 상위 5개의 유사한 항목을 검색
    webtoon_distances, webtoon_indices = webtoon_index.search(np.array([query_embedding]), k)


    return {"webtoon_key": webtoon_indices[0].tolist()}


app = FastAPI()
data_config = DataIngestionConfig()

webtoon, rating, anime = get_data(data_config)
anime_embeddings, webtoon_embeddings = get_embeddings(data_config)

model = SentenceTransformer('paraphrase-distilroberta-base-v1')
model.max_seq_length = 384

webtoon_index = faiss.IndexFlatL2(webtoon_embeddings.shape[1])
webtoon_index.add(webtoon_embeddings)

animation_index = faiss.IndexFlatL2(anime_embeddings.shape[1])
animation_index.add(anime_embeddings)


@app.post("/query-recommend", response_model=QueryRecOut)
def query_recommend_webtoon(user_data: QueryRecIn):
    query = user_data.query
    try:
        items = get_response_only_query(query)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid inputs")
    
    results = {"webtoon_key": items["webtoon_key"]}
    
    return results

@app.post("/user-recommend", response_model=UserOut)
def user_recommend_webtoon(user_data: UserIn):
    user_id = user_data.user_id
    query = user_data.query
    logging.info(f"user_id {user_id} and query {query}")
    
    try:
        items = get_response_with_query(user_id, query)
        logging.info(f"items {items}")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid inputs")
    
    results = {"anime_key": items["anime_key"], "webtoon_key": items["webtoon_key"]}
    
    return results

@app.post("/anime", response_model=AnimeOut)
def anime_data(user_id_list:AnimeIn):
    try:
        id_list = user_id_list.user_id_list
    
        item = anime[anime["MAL_ID"].isin(id_list)].to_dict()
        results = {"anime_data": item}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid inputs")
    
    return results
    
@app.post("/webtoon", response_model=WebtoonOut)
def webtoon_data(webtoon_id_list:WebtoonIn):
    try:
        id_list = webtoon_id_list.webtoon_id_list

        item = webtoon[webtoon["id"].isin(id_list)].to_dict()
        # print(item)
        results = {"webtoon_data": item}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid inputs")
    return results
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5020)

            



    

