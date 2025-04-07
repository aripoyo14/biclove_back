from pydantic import BaseModel, constr, ConfigDict
from typing import Annotated, List, Optional
import datetime

# 共通の datetime エンコーダ
DATETIME_ENCODER = {datetime.datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")}


# -------------------------------
# User スキーマ
# -------------------------------
class User(BaseModel):
    id: Optional[int] = None
    name: Annotated[str, constr(min_length=1)]
    affiliation: Annotated[str, constr(min_length=1)]
    email: Annotated[str, constr(min_length=1)]
    password_hash: Annotated[str, constr(min_length=1)]
    total_thanks: int = 0
    total_view: int = 0
    created_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(json_encoders=DATETIME_ENCODER)


# -------------------------------
# Meeting スキーマ
# -------------------------------
class Meeting(BaseModel):
    id: Optional[int] = None
    user_id: int
    title: Annotated[str, constr(min_length=1)]
    summary: Annotated[str, constr(min_length=1)]
    created_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(json_encoders=DATETIME_ENCODER)


# -------------------------------
# Knowledge スキーマ
# -------------------------------
class Knowledge(BaseModel):
    id: Optional[int] = None
    user_id: int
    meeting_id: int
    title: Annotated[str, constr(min_length=1)]
    content: Annotated[str, constr(min_length=1)]
    thanks_count: int = 0
    created_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(json_encoders=DATETIME_ENCODER)


# -------------------------------
# Challenge スキーマ
# -------------------------------
class Challenge(BaseModel):
    id: Optional[int] = None
    user_id: int
    meeting_id: int
    title: Annotated[str, constr(min_length=1)]
    content: Annotated[str, constr(min_length=1)]
    created_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(json_encoders=DATETIME_ENCODER)


# -------------------------------
# SolutionKnowledge リクエストスキーマ
# -------------------------------
class SolutionKnowledgeRequest(BaseModel):
    content: Annotated[str, constr(min_length=1)]


# -------------------------------
# Thanks スキーマ
# -------------------------------
class Thanks(BaseModel):
    id: Optional[int] = None
    user_id: int
    knowledge_id: int
    created_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(json_encoders=DATETIME_ENCODER)


# -------------------------------
# View スキーマ
# -------------------------------
class View(BaseModel):
    id: Optional[int] = None
    user_id: int
    knowledge_id: int
    created_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(json_encoders=DATETIME_ENCODER)


# -------------------------------
# FastAPI レスポンス用 スキーマ
# -------------------------------

class MeetingResponse(BaseModel):
    id: Optional[int] = None
    user_id: int
    title: str
    summary: str
    created_at: Optional[datetime.datetime] = None
    challenges: List[Challenge] = []
    knowledges: List[Knowledge] = []

    model_config = ConfigDict(json_encoders=DATETIME_ENCODER)


# RAG / Pinecone 用マッチスキーマ
class Match(BaseModel):
    id: str
    text: str


# SolutionKnowledge エンドポイントレスポンス
class SolutionKnowledgeResponse(BaseModel):
    summary: str  # 複数のナレッジを要約したテキスト
    knowledges: List[dict] = []  # ナレッジの詳細情報（title, content, user_id, user_name）