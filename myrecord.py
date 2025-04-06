from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, constr, ConfigDict
import requests
import json
from typing import Annotated
import datetime

from db_control import crud, mymodels_MySQL

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA
# from langchain_community.llms import OpenAI
from pinecone import Pinecone, ServerlessSpec

from typing import List, Optional
from openai import OpenAI

import numpy as np
import pandas as pd
import os

# FastAPIアプリの初期化
app = FastAPI()

# OpenAIクライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["https://tech0-gen-9-step3-1-node-13.azurewebsites.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MySQLのテーブル作成し、アプリケーション初期化時にテーブルを作成
from db_control.create_tables_MySQL import init_db
init_db()

# 環境変数の設定
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Pineconeの初期化
pc = Pinecone(api_key=pinecone_api_key)
index_name = "biclove-flowledge"

# インデックスの確認または作成
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # 使用するモデルの次元数
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
dense_index = pc.Index(index_name)

# ベクトル生成モデルの初期化
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

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

class solutionknowledge(BaseModel):
    content: Annotated[str, constr(min_length = 1)]
    
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

# RAG用のモデル
class Match(BaseModel):
    id: str
    text: str

# RAG用のモデル
class SolutionKnowledgeResponse(BaseModel):
    summary: str #複数のナレッジを要約したテキスト
    knowledges: List[dict] = [] #ナレッジの詳細情報（title, content, user_id, user_name）
    
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
    
@app.post("/solution_knowledge", response_model=SolutionKnowledgeResponse)
def create_solution_knowledge(challenge: solutionknowledge):
    """
    指定されたナレッジの内容をもとに、解決策を生成するエンドポイント
    """
    try:
        # challengeの内容をベクトル化
        challenge_vector = embeddings.embed_query(challenge.content)
        
        # ベクトル検索
        response = dense_index.query(
            top_k=5,
            include_metadata=True,
            vector=challenge_vector
        )
        
        # 取得したナレッジを要約する
        knowledge_texts = [match["metadata"]["text"] for match in response.get("matches", [])]
        combined_knowledge = "\n".join(knowledge_texts)
        
        # 要約を生成
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Please summarize the following text: {combined_knowledge}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        summary = summary_response.choices[0].message.content
        
        # ナレッジIDを取得
        knowledge_ids = [int(match["id"]) for match in response.get("matches", [])]
        
        # ナレッジの詳細情報を取得
        knowledges = crud.get_knowledge_details(knowledge_ids)
        
        # サマリーとidとナレッジの詳細情報を返す
        return {
            "summary": summary,
            "knowledges": knowledges
        }
                
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    



