import os
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    anime_data_path: str = os.path.join("data", "anime_merged.csv")
    rating_data_path: str = os.path.join("data", "rating_complete.csv")
    webtoon_data_path: str = os.path.join("data", "Webtoon Dataset.csv")
    anime_embedding_path: str = os.path.join("data", "anime_embeddings.npy")
    webtoon_embedding_path: str = os.path.join("data", "webtoon_embeddings.npy")
