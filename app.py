from typing import Any
import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from config import DataIngestionConfig

from utils import *

class UserData(BaseModel):
    user_data: int

class Recommendation(BaseModel):
    anime_key: list
    webtoon_key: list
    webtoon_score: list
    
class AnimeData(BaseModel):
    user_id_list: list

class Animation(BaseModel):
    anime_data: Any

class WebtoonData(BaseModel):
    webtoon_id_list: list
    
class Webtoon(BaseModel):
    webtoon_data: Any

def get_data(config:DataIngestionConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    webtoon_data = pd.read_csv(config.webtoon_data_path)
    rating_data = pd.read_csv(config.rating_data_path)
    anime_data = pd.read_csv(config.anime_data_path)
    
    return webtoon_data, rating_data, anime_data

def get_embeddings(config:DataIngestionConfig) -> tuple[np.array, np.array]:
    anime_embeddings = np.load(config.anime_embedding_path)
    webtoon_embeddings = np.load(config.webtoon_embedding_path)
    
    return anime_embeddings, webtoon_embeddings


def get_response(user_id:int):
    preferred_animes = get_preferred_anime_from_user(user_id, rating)[:2] # 2 is used to reduce recommenation space
    
    top_itmes = get_recommenations(webtoon, anime, preferred_animes, anime_embeddings, webtoon_embeddings)
    return top_itmes
    

app = FastAPI()
data_config = DataIngestionConfig()

webtoon, rating, anime = get_data(data_config)
anime_embeddings, webtoon_embeddings = get_embeddings(data_config)

@app.post("/recommend", response_model=Recommendation)
def recommend_webtoon(user_data: UserData):
        
    data = user_data.user_data
    items = get_response(data)
    # print(items)
    results = {"anime_key": items["anime_key"], 
               "webtoon_key": items["webtoon_key"], 
               "webtoon_score": items["webtoon_score"]}
    
    return results

@app.post("/anime", response_model=Animation)
def anime_data(user_id_list:AnimeData):
    id_list = user_id_list.user_id_list
    
    item = anime[anime["MAL_ID"].isin(id_list)].to_dict()
    results = {"anime_data": item}
    return results
    
@app.post("/webtoon", response_model=Webtoon)
def webtoon_data(webtoon_id_list:WebtoonData):
    id_list = webtoon_id_list.webtoon_id_list

    item = webtoon[webtoon["id"].isin(id_list)].to_dict()
    # print(item)
    results = {"webtoon_data": item}
    return results
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5020)
            
            



    

