from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from db_control import crud
from db_control.create_tables_MySQL import init_db
from schemas import (
    SolutionKnowledgeRequest,
    SolutionKnowledgeResponse,
    MeetingResponse
)

import os
import datetime
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# FastAPI アプリの初期化
app = FastAPI()

# OpenAI クライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CORS ミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MySQL のテーブル作成（アプリ起動時に実行）
init_db()

# 環境変数の取得
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Pinecone の初期化
pc = Pinecone(api_key=pinecone_api_key)
index_name = "biclove-flowledge"

# インデックスの確認または作成
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# ベクトル生成モデルの初期化
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


# -------------------------------
# ルーティング
# -------------------------------

@app.get("/")
def index():
    return {"message": "FastAPI Arichan part!"}


@app.get("/latest_meeting", response_model=List[MeetingResponse])
def get_latest_meeting(user_id: int = Query(..., description="ユーザーID")):
    """
    指定されたユーザーIDの最新の会議データと関連するチャレンジとナレッジを取得するエンドポイント
    """
    try:
        result = crud.get_meeting_with_related_data_using_join_optimized(user_id=user_id, limit=4)
        return result
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return []

@app.post("/solution_knowledge", response_model=SolutionKnowledgeResponse)
def create_solution_knowledge(challenge: SolutionKnowledgeRequest):
    """
    指定されたナレッジの内容をもとに、解決策を生成するエンドポイント
    """
    try:
        # チャレンジ内容をベクトル化
        challenge_vector = embeddings.embed_query(challenge.content)

        # Pinecone によるベクトル検索
        response = index.query(
            top_k=5,
            include_metadata=True,
            vector=challenge_vector
        )

        # ナレッジをまとめて要約
        knowledge_texts = [match["metadata"]["text"] for match in response.get("matches", [])]
        combined_knowledge = "\n".join(knowledge_texts)

        # GPT による要約生成
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

        # ナレッジ ID を取得
        knowledge_ids = [int(match["id"]) for match in response.get("matches", [])]

        # ナレッジの詳細情報を取得
        knowledges = crud.get_knowledge_details(knowledge_ids)

        return {
            "summary": summary,
            "knowledges": knowledges
        }

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=str(e))
