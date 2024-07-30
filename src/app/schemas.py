from typing import Any, Optional
from pydantic import BaseModel

class UserIn(BaseModel):
    user_id: int
    query: str
    
class UserOut(BaseModel):
    anime_key: list
    webtoon_key: list

class QueryRecIn(BaseModel):
    query: Optional[str]

class QueryRecOut(BaseModel):
    webtoon_key: list

class AnimeIn(BaseModel):
    user_id_list: list

class AnimeOut(BaseModel):
    anime_data: Any

class WebtoonIn(BaseModel):
    webtoon_id_list: list
    
class WebtoonOut(BaseModel):
    webtoon_data: Any