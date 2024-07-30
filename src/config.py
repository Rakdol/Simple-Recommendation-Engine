import os
import sys
from pathlib import Path
from dataclasses import dataclass


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
# sys.path.append(str(PACKAGE_ROOT))
data_path= str(PACKAGE_ROOT) + "/data"

@dataclass
class DataIngestionConfig:
    anime_data_path: str = os.path.join(data_path, "anime_merged.csv")
    rating_data_path: str = os.path.join(data_path, "rating_complete.csv")
    webtoon_data_path: str = os.path.join(data_path, "Webtoon Dataset.csv")
    anime_embedding_path: str = os.path.join(data_path, "anime_embeddings.npy")
    webtoon_embedding_path: str = os.path.join(data_path, "webtoon_embeddings.npy")
