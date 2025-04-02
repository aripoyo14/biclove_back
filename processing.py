from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, constr
import requests
import json
from typing import Annotated
import datetime

from db_control import crud, mymodels_MySQL

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from pinecone import Pinecone, ServerlessSpec

from typing import List, Optional
import openai

import numpy as np
import os

# FastAPI アプリの初期化
app = FastAPI()

# OpenAI クライアントの作成
client = OpenAI()

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
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Pinecone の初期化
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = "knowledge-index"

# インデックスの確認または作成
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # 使用するモデルの次元数
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# ベクトル生成モデルの初期化 (OpenAI Embeddings を使用)
model = OpenAIEmbeddings(model="text-embedding-ada-002")


class Meeting(BaseModel):
    id: int | None = None
    user_id: str
    title: str
    summary: str
    knowledge: str
    issues: str
    solutionKnowledge: str
    created_at: datetime.datetime | None = None


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI server!"}


@app.post("/meeting")
def post_finalized_meeting(newMeeting: Meeting):
    if newMeeting.created_at is None:
        newMeeting.created_at = datetime.datetime.now()
        
    # 🔥 1. Knowledge と Issues のベクトル化
    knowledge_vector = model.embed_query(newMeeting.knowledge)
    issues_vector = model.embed_query(newMeeting.issues)

    # 🔥 2. Pinecone にベクトルを保存
    index.upsert([
        (f"knowledge-{newMeeting.id}", knowledge_vector, {"text": newMeeting.knowledge, "type": "knowledge"}),
        (f"issues-{newMeeting.id}", issues_vector, {"text": newMeeting.issues, "type": "issues"})
    ])


    # 🔥 3. Pinecone で類似検索を実行 (Issues に対する Knowledge の検索)
    response = index.query(
        vector=issues_vector,
        top_k=5,
        include_metadata=True
    )
    
    # 🔥 4. 検索結果を要約する
    knowledge_texts = [match['metadata']['text'] for match in response['matches']]
    combined_knowledge = "\n".join(knowledge_texts)
    
    # OpenAI API を使って要約する
    prompt = f"""
    以下の情報を要約してください。内容を簡潔にまとめ、主要なポイントを抽出してください。

    {combined_knowledge}
    """
    
    try:
        summary_response = openai.ChatCompletion.create(
            model="gpt-4",  # モデルを指定
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        summarized_text = summary_response.choices[0].message["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")

    # 🔥 5. MySQL に保存 (タイトルや要約など)
    try:
        db = SessionLocal()
        insert_query = text("""
            INSERT INTO meetings (id, user_id, title, summary, solutionKnowledge, created_at)
            VALUES (:id, :user_id, :title, :summary, :solutionKnowledge, :created_at)
        """)
        db.execute(insert_query, {
            "id": newMeeting.id,
            "user_id": newMeeting.user_id,
            "title": newMeeting.title,
            "summary": newMeeting.summary,
            "knowledge": newMeeting.knowledge,
            "issues": newMeeting.issues,
            "solutionSummary": summarized_text,
            "solutionKnowledge": newMeeting.solutionKnowledge,
            "created_at": newMeeting.created_at
        })
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"MySQL Error: {str(e)}")
    finally:
        db.close()

    print(f"受け取ったデータ： {newMeeting}")
    
    stats = index.describe_index_stats()
    print(stats)
    
    return {"res": "ok", "ID": newMeeting.id}


# @app.post("/add_knowledge/")
# def add_knowledge(content: str, db = Depends(get_db)):
    # ナレッジをベクトル化
#    vector = model.encode(content).astype(np.float32)
#    index.add(np.array([vector]))  # FAISS に追加

    # ベクトルをバイトデータに変換して保存
#    vector_data = vector.tobytes()
#    db.execute(
#        text("INSERT INTO knowledge (content, vector) VALUES (:content, :vector)"),
#        {"content": content, "vector": vector_data}
#    )
#    db.commit()
#    return {"status": "Knowledge added successfully"}

#@app.post("/search_knowledge/")
#def search_knowledge(query: str):
#    query_vector = model.encode(query).astype(np.float32)
#    D, I = index.search(np.array([query_vector]), k=5)  # 上位5件を検索

#    results = []
#    for idx in I[0]:
#        if idx == -1:
#            continue
#        results.append({"index": int(idx), "score": float(D[0][0])})
    
#    return {"results": results}

#@app.post("/generate_response/")
#def generate_response(query: str):
#    response = openai.ChatCompletion.create(
#        model="gpt-4",
#        messages=[
#            {"role": "system", "content": "You are a helpful assistant."},
#            {"role": "user", "content": query},
#        ]
#    )
#    return {"response": response.choices[0].message["content"]}






# 修正後のミーティングの内容をDBに登録（ナレッジとチャレンジはベクトル化）


# DBに登録されたベクトル化されたチャレンジに対して過去のナレッジを検索

# ナレッジをレコメンデーションする応答文と参照元となるナレッジを取得

# レコメンデーションの応答文と参照元となるナレッジをDBに保存




