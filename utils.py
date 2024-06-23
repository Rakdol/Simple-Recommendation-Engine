import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


def get_preferred_anime_from_user(user_id: int, rating_complete: pd.DataFrame) -> pd.DataFrame:
    return rating_complete[(rating_complete['user_id'] == user_id) & (rating_complete['rating'] >= 9)]['anime_id'].tolist()


def get_simliar_animes(anime:pd.DataFrame, anime_id:int, embeddings:np.array, k:int) -> list:
    try:
        idx = anime[anime["MAL_ID"] == anime_id] .index[0]
    except:
        return []
    
    cosine_sim_matrix = cosine_similarity(embeddings)
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    
    # Sort the animes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and scores of the k most similar animes
    k_indices_scores = sim_scores[1:k+1]

    # Separate indices and scores
    k_indices = [i[0] for i in k_indices_scores]
    k_scores = [round(i[1], 3) for i in k_indices_scores]

    # Get the MAL_IDs of the k most similar animes
    k_anime_ids = anime['MAL_ID'].iloc[k_indices].tolist()

    return list(zip(k_anime_ids, k_scores))


def get_relevant_webtoons(webtoon:pd.DataFrame, anime:pd.DataFrame, preferred_anime_id:list, anime_embeddings:np.array, webtoon_embeddings:np.array, k:int=3) -> list:
    recommendations = {}
    simliar_animess = get_simliar_animes(anime, preferred_anime_id, anime_embeddings, k=k)
    anime_toon_matrix = cosine_similarity(anime_embeddings, webtoon_embeddings)
    for anime_id, cosine_score in simliar_animess:
        try:
            sim_scores = list(enumerate(anime_toon_matrix[anime_id]))
        except:
            continue
        
        # Sort the animes based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        k_indices_scores = sim_scores[1:k+1]
        
        # Separate indices and scores
        k_indices = [i[0] for i in k_indices_scores]
        k_scores = [round(i[1], 3) for i in k_indices_scores]
        
        # Get the webtoon id's of the k most similar animes
        k_webtoon_ids = webtoon['id'].iloc[k_indices].tolist()
        recommendations[anime_id] = list(zip(k_webtoon_ids, k_scores))
    return recommendations


def get_recommenations(webtoon:pd.DataFrame, anime:pd.DataFrame, preferred_animes:list, anime_embeddings:np.array, webtoon_embeddings:np.array, k:int=3) -> dict:
    recommendations = []
    for preferred_anime_id in preferred_animes:
        recommendations.append(get_relevant_webtoons(webtoon, anime, preferred_anime_id, anime_embeddings, webtoon_embeddings))
    
    anime_key = []
    webtoon_key = []
    webtoon_score = []
    for rec in recommendations[:2]:
        for key, vals in rec.items():
            anime_key.append(key)
            for val in vals:
                webtoon_key.append(val[0])
                webtoon_score.append(val[1])
                
    webtoon_score = [float(score) for score in webtoon_score]
    anime_key = [int(animal) for animal in anime_key]
    webtoon_key = [int(webt) for webt in webtoon_key]

    return {"anime_key": anime_key, "webtoon_key": webtoon_key, "webtoon_score": webtoon_score}