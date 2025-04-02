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
from langchain.llms import OpenAI

from typing import List, Optional
import openai

from sentence_transformers import SentenceTransformer
import faiss

import numpy as np
import os


# FastAPI アプリの初期化
app = FastAPI()

# MySQLのテーブル作成
from db_control.create_tables_MySQL import init_db
# アプリケーション初期化時にテーブルを作成
init_db()

# ベクトルモデルのロード
model = SentenceTransformer('all-mpnet-base-v2')
# FAISS インデックスの作成 (ベクトル次元数を指定)
index = faiss.IndexFlatL2(768)  # 768次元のベクトル用
# GPT-4 モデル (APIキーを設定)
openai.api_key = os.getenv("OpenAI_API_KEY") 

class Meeting(BaseModel):
    id: int | None = None
    user_id: str
    title: str
    summary: str
    issues: str
    knowledge: str
    solutionKnowledge: str
    created_at: datetime.datetime | None = None

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
def read_root():
    return {"message": "Welcome to the FastAPI server!"}

@app.post("/meeting")
def post_finalized_meeting(newMeeting: Meeting):
    if newMeeting.created_at is None:
        newMeeting.created_at = datetime.datetime.now()

    print(f"受け取ったデータ： {newMeeting}")
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




