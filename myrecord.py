from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, constr, ConfigDict
import requests
import json
from typing import Annotated, List, Optional
import datetime

from db_control import crud, mymodels_MySQL
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select

# モデルクラスのインポート
from db_control.mymodels_MySQL import (
    User,
    Meeting,
    Knowledge,
    Challenge,
    Thanks,
    View,
    SolutionKnowledge,
    Reference
)

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pinecone import Pinecone, ServerlessSpec

import openai

import numpy as np
import os
import pandas as pd

# 環境変数の設定
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# FastAPI アプリの初期化
app = FastAPI()

# OpenAI クライアントの作成
client = OpenAI()

# Pinecone の初期化
pc = Pinecone(api_key=pinecone_api_key)
index_name = pinecone_index_name

# インデックスの確認または作成
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # 使用するモデルの次元数
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

dense_index = pc.Index(index_name)

# ベクトル生成モデルの初期化 (OpenAI Embeddings を使用)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["https://tech0-gen-9-step3-1-node-13.azurewebsites.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# レスポンスモデルの定義
class MeetingResponse(BaseModel):
    id: Optional[int] = None
    user_id: int
    title: str
    summary: str
    time: datetime.time
    created_at: Optional[datetime.datetime] = None
    model_config = ConfigDict(from_attributes=True)

class ChallengeRequest(BaseModel):
    content: str

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
    
@app.post("/generate_knowledge")
def generate_knowledge(challenge: ChallengeRequest):
    """
    指定されたチャレンジの内容に基づいて類似度検索を行い、関連するナレッジを生成するエンドポイント
    
    Args:
        challenge (ChallengeRequest): チャレンジの内容
    
    Returns:
        dict: 生成された解決策と参照元ナレッジの情報
    """
    try:
        print("✅ チャレンジの内容:", challenge.content)
        
        # チャレンジの内容をベクトル化
        challenge_vector = embeddings.embed_query(challenge.content)
        print("✅ ベクトル化完了:", challenge_vector)
        
        # Pineconeで類似度検索を実行
        print("✅ Pinecone検索開始")
        response = dense_index.query(
            vector=challenge_vector,
            top_k=5,
            include_metadata=True
        )
        print(f"✅ 類似検索完了",response)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone 検索エラー: {str(e)}")


        
        
    # except Exception as e:
    #     print(f"❌ エラーが発生しました: {str(e)}")
    #     print(f"❌ エラーの型: {type(e)}")
    #     raise HTTPException(status_code=500, detail=f"ナレッジ生成エラー: {str(e)}")

        
    

