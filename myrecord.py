from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr, ConfigDict
import requests
import json
from db_control import crud, mymodels_MySQL
from typing import Annotated, List
import datetime
import pandas as pd
from sqlalchemy.sql import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# MySQLのテーブル作成
from db_control.create_tables_MySQL import init_db

# アプリケーション初期化時にテーブルを作成
init_db()

# スキーマ定義
class User(BaseModel):
    id: int | None = None
    name: Annotated[str, constr(min_length = 1)]
    affiliation: Annotated[str, constr(min_length = 1)]
    email: Annotated[str, constr(min_length = 1)]
    password_hash: Annotated[str, constr(min_length = 1)]
    total_thanks: int = 0
    total_view: int = 0
    created_at: datetime.datetime | None = None
    model_config = ConfigDict(json_encoders={datetime.datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")})

class Meeting(BaseModel):
    id: int | None = None
    user_id: int
    title: Annotated[str, constr(min_length = 1)]
    summary: Annotated[str, constr(min_length = 1)]
    time: datetime.time
    created_at: datetime.datetime | None = None
    model_config = ConfigDict(json_encoders={datetime.datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")})

class Knowledge(BaseModel):
    id: int | None = None
    user_id: int
    meeting_id: int
    title: Annotated[str, constr(min_length = 1)]
    content: Annotated[str, constr(min_length = 1)]
    thanks_count: int = 0
    created_at: datetime.datetime | None = None
    model_config = ConfigDict(json_encoders={datetime.datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")})

class Challenge(BaseModel):
    id: int | None = None
    user_id: int
    meeting_id: int
    title: Annotated[str, constr(min_length = 1)]
    content: Annotated[str, constr(min_length = 1)]
    created_at: datetime.datetime | None = None
    model_config = ConfigDict(json_encoders={datetime.datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")})

class Tag(BaseModel):
    id: int | None = None
    name: Annotated[str, constr(min_length = 1)]

class Thanks(BaseModel):
    id: int | None = None
    user_id: int
    knowledge_id: int
    created_at: datetime.datetime | None = None
    model_config = ConfigDict(json_encoders={datetime.datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")})

class View(BaseModel):
    id: int | None = None
    user_id: int
    knowledge_id: int
    created_at: datetime.datetime | None = None
    model_config = ConfigDict(json_encoders={datetime.datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")})

# レスポンスモデルの定義
class MeetingResponse(BaseModel):
    id: int | None = None
    user_id: int
    title: str
    summary: str
    time: datetime.time
    created_at: datetime.datetime | None = None
    challenges: List[Challenge] = []
    knowledges: List[Knowledge] = []
    model_config = ConfigDict(json_encoders={datetime.datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")})

app = FastAPI()

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["https://tech0-gen-9-step3-1-node-13.azurewebsites.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {"message": "FastAPI Arichan part!"}

@app.get("/latest_meeting", response_model=List[MeetingResponse])
def get_latest_meeting(user_id: int = Query(..., description="ユーザーID")):
    """
    指定されたユーザーIDの最新の会議データと関連するチャレンジとナレッジを取得するエンドポイント
    
    Args:
        user_id (int): ユーザーID
    
    Returns:
        list: 会議データと関連するチャレンジとナレッジのリスト
    """
    try:
        # JOINを使用する関数を使用して特定ユーザーの最新の会議データと関連するチャレンジとナレッジを取得
        result = crud.get_meeting_with_related_data_using_join_optimized(user_id=user_id, limit=4)
        return result
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return []

