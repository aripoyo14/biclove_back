from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr
from typing import Annotated, List, Optional

# DB関連のインポート
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from db_control import mymodels_MySQL, crud 
from db_control.connect_MySQL import engine
from db_control.create_tables_MySQL import init_db
from db_control.crud import (
    get_meeting_with_related_data_using_join_optimized,
    get_knowledge_details,
    create_index,
    get_all_vectors,
    save_meeting_with_knowledge_and_challenge,
    get_meeting_summary,
    get_meeting_knowledges
)

# スキーマ関連のインポート
from schemas import (
    SolutionKnowledgeRequest,
    SolutionKnowledgeResponse,
    MeetingResponse,
    OtherMeetingResponse
)

# OpenAI関連のインポート
from openai import OpenAI
from langchain_openai import OpenAI as LangChainOpenAI
from langchain_openai import OpenAIEmbeddings

# ベクトルデータベース関連のインポート
from pinecone import Pinecone, ServerlessSpec

# ユーティリティ関連のインポート
import os
from dotenv import load_dotenv
import datetime
import re
import traceback
import hashlib
import time
import numpy as np
import requests
import json

# SessionLocal を定義（connect_MySQL.py の engine を利用）
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# .env ファイルの読み込み（環境変数を使うため）
load_dotenv()

# OpenAI APIキーを環境変数から取得して設定
api_key = os.getenv("OPENAI_API_KEY")

# 環境変数が正しく設定されているかチェック
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment variables.")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 環境変数の取得
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

# 環境変数のバリデーション
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
    raise RuntimeError("Pinecone configuration is missing in environment variables")

# Pineconeクライアントの初期化
pc = Pinecone(api_key=PINECONE_API_KEY)


# インデックスの確認または作成
try:
    # インデックスが存在するか確認
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        # インデックスが存在しない場合は作成
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # OpenAI text-embedding-ada-002 の次元数
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
    # インデックスの取得
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"❌ Pineconeの初期化エラー: {str(e)}")
    raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

# ベクトル生成モデルの初期化
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

#DBチェック
for key in ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]:
    if not os.getenv(key):
        raise RuntimeError(f"Missing database config: {key}")

# FastAPI アプリの初期化
app = FastAPI()

# CORS ミドルウェアの設定（フロントエンドと通信を可能にする）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # セキュリティを考慮して本番環境では制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MySQL のテーブル作成（アプリ起動時に実行）※役割調べないといけない
init_db()

# アップロードされた音声ファイルを保存するディレクトリを作成
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# OpenAI Whisperを使って音声を文字起こしする関数（★新方式に書き換え）
async def transcribe_audio(file_path: str):
    try:
        with open(file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")

# OpenAI GPTを使って文字起こしデータを要約する関数
def generate_summary(text: str):
    try:
        prompt = f"""
        次の文章を、以下の3つのカテゴリに整理してください。
        
        ①【会議の要約】
        → **200文字以内** で、簡潔に会議の内容をまとめてください。
        
        ②【知見】
        → **新しい発見や重要な情報** を箇条書きでリストにしてください。
        
        ③【悩み】
        → **問題点や課題** を箇条書きでリストにしてください。
        
        文章: {text}
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional summarizer."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation error: {e}")

# GPTでタイトルを生成
def generate_title(text: str) -> str:
    try:
        prompt = f"""
        以下の文章は会議の内容です。この会議にふさわしいタイトルを、日本語で10〜30文字以内で1つだけ提案してください。
        文章: {text}
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "あなたはタイトル生成の専門家です。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title generation error: {e}")

# knowledgeのタイトルをGPTで生成
def generate_knowledge_title(content: str) -> str:
    try:
        prompt = f"""
以下の内容は会議から得られた知見の一つです。この知見にふさわしいタイトルを「日本語で」「10〜30文字以内」で1つだけ提案してください。
内容: {content}
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "あなたは知見タイトル生成のプロです。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print("❌ 知見タイトル生成エラー:", e)
        return "自動生成知見タイトル"

# challengeのタイトルをGPTで生成
def generate_challenge_title(content: str) -> str:
    try:
        prompt = f"""
以下の内容は会議で議論された課題の一つです。この課題にふさわしいタイトルを「日本語で」「10〜30文字以内」で1つだけ提案してください。
内容: {content}
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "あなたは課題タイトル生成のプロです。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print("❌ 課題タイトル生成エラー:", e)
        return "自動生成課題タイトル"

# GPT出力のパース関数（要約 / 知見 / 悩み に分ける）
def parse_summary_response(summary_text: str) -> dict:
    try:
        summary_match = re.search(r"①【会議の要約】\n?(.*?)(?:\n②|$)", summary_text, re.DOTALL)
        knowledge_match = re.search(r"②【知見】\n?(.*?)(?:\n③|$)", summary_text, re.DOTALL)
        challenge_match = re.search(r"③【悩み】\n?(.*)", summary_text, re.DOTALL)

        summary = summary_match.group(1).strip() if summary_match else ""

        def parse_items(block_text):
            items = []
            blocks = re.split(r"-\s*", block_text.strip())
            for block in blocks:
                if not block.strip():
                    continue
                title_match = re.search(r"タイトル[:：](.+)", block)
                content_match = re.search(r"内容[:：](.+)", block, re.DOTALL)
                title = title_match.group(1).strip() if title_match else "タイトル不明"
                content = content_match.group(1).strip() if content_match else block.strip()
                items.append({"title": title, "content": content})
            return items

        knowledges = parse_items(knowledge_match.group(1)) if knowledge_match else []
        challenges = parse_items(challenge_match.group(1)) if challenge_match else []

        return {
            "summary": summary,
            "knowledges": knowledges,
            "challenges": challenges,
        }

    except Exception as e:
        raise ValueError(f"GPT出力のパースに失敗しました: {e}")

# -------------------------------
# ルーティング
# -------------------------------

@app.get("/")
def home():
    return {"message": "Thank you for attending biclove-flowledge!"}

# 音声ファイルを受け取り、文字起こしし、データベースに保存するエンドポイント
@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    user_id: int = Form(...)
):
    try:
        # 音声ファイルをサーバーに保存
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 文字起こし処理
        transcript = await transcribe_audio(file_path)
        
        # 要約生成とパース
        summary_text = generate_summary(transcript)
        parsed = parse_summary_response(summary_text)
        title = generate_title(parsed["summary"])
        
        # データベースへの登録
        db: Session = SessionLocal()
        try:
            # 会議、知見、課題を保存
            response_data = save_meeting_with_knowledge_and_challenge(
                db=db,
                user_id=user_id,
                title=title,
                summary=parsed["summary"],
                knowledges=[
                    {
                        "title": generate_knowledge_title(item["content"]),
                        "content": item["content"]
                    }
                    for item in parsed["knowledges"]
                ],
                challenges=[
                    {
                        "title": generate_challenge_title(item["content"]),
                        "content": item["content"]
                    }
                    for item in parsed["challenges"]
                ]
            )
            
            # トランスクリプトを追加
            response_data["transcript"] = transcript
            response_data["message"] = "Meeting + 知見 + 悩み 登録完了"
            
            return response_data
        finally:
            db.close()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# 会議の文字起こしデータを取得し、要約を生成するエンドポイント(ORM版)
@app.get("/get-summary/{meeting_id}")
async def get_summary(meeting_id: int):
    db: Session = SessionLocal()
    try:
        return get_meeting_summary(db, meeting_id)
    finally:
        db.close()

@app.get("/get-knowledge/{meeting_id}")
async def get_knowledge(meeting_id: int):
    db: Session = SessionLocal()
    try:
        return get_meeting_knowledges(db, meeting_id)
    finally:
        db.close()

# フロント側で編集された会議情報を更新するエンドポイント
@app.put("/update-meeting/{meeting_id}")
async def update_meeting(meeting_id: int, data: dict = Body(...)):
    """
    フロント側で編集された会議情報、知見、悩みの内容を更新し、
    更新された知見についてはベクトルDB（例：Pinecone）にも登録する。
    
    Args:
        meeting_id (int): 更新対象の会議ID
        data (dict): 更新対象のデータ {"title": ..., "summary": ...,
                  "knowledges": [{"id": ..., "title": ..., "content": ...}, ...],
                  "challenges": [{"id": ..., "title": ..., "content": ...}, ...]}
    """
    db: Session = SessionLocal()
    try:
        # 1. 会議情報の取得と更新
        meeting = db.query(mymodels_MySQL.Meeting).filter_by(id=meeting_id).first()
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")

        meeting.title = data.get("title", meeting.title)
        meeting.summary = data.get("summary", meeting.summary)
        
        # 2. 更新済み知見をまとめるリスト
        updated_knowledges = []
        
        # 3. 知見の更新
        for k in data.get("knowledges", []):
            knowledge = db.query(mymodels_MySQL.Knowledge).filter_by(id=k["id"]).first()
            if knowledge:
                knowledge.title = k.get("title", knowledge.title)
                knowledge.content = k.get("content", knowledge.content)
                db.add(knowledge)
                updated_knowledges.append({
                    "id": knowledge.id,
                    "content": knowledge.content
                })
        
        # 4. 悩み（Challenge）の更新
        for c in data.get("challenges", []):
            challenge = db.query(mymodels_MySQL.Challenge).filter_by(id=c["id"]).first()
            if challenge:
                challenge.title = c.get("title", challenge.title)
                challenge.content = c.get("content", challenge.content)
                db.add(challenge)
        
        # 5. すべての更新を一括コミット
        db.commit()
        db.refresh(meeting)
        
        # 6. 更新済み知見をまとめてベクトル化処理
        if updated_knowledges:
            create_index(updated_knowledges, index, embeddings)
        
        return {
            "message": "Meeting内容を更新しベクトル化しました",
            "meeting_id": meeting_id
        }
    finally:
        db.close()

@app.get("/vectors")
async def get_all_vectors_endpoint():
    return get_all_vectors(index)

@app.get("/latest_meeting", response_model=List[MeetingResponse])
def get_latest_meeting(user_id: int = Query(..., description="ユーザーID")):
    """
    指定されたユーザーIDの最新の会議データと関連するチャレンジとナレッジを取得するエンドポイント
    """
    try:
        result = get_meeting_with_related_data_using_join_optimized(user_id=user_id, limit=4)
        return result
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return []

@app.get("/latest_meeting/other_users", response_model=List[OtherMeetingResponse])
def get_latest_meeting_other_users(
    user_id: int = Query(..., description="除外するユーザーID"),
    limit: int = Query(4, description="取得する会議の数")
):
    """
    指定されたユーザーID以外の最新の会議データと関連するチャレンジとナレッジを取得するエンドポイント
    """
    try:
        result = get_meeting_with_related_data_using_join_optimized(user_id=user_id, limit=limit, exclude_user=True)
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
        print(response)
        knowledge_texts = [match["metadata"]["text"] for match in response.get("matches", [])]
        
        print("----------",knowledge_texts)
        
        combined_knowledge = "\n".join(knowledge_texts)
        print(combined_knowledge)
        
        # GPT による要約生成
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"""
        以下のユーザーの質問に対する回答をコンテキスト情報を元に500文字以内で生成してください。コンテキスト情報に含まれない内容は回答しないでください。
        ユーザーの質問：{challenge.content}
        コンテキスト：{combined_knowledge}
        """}
            ],
            temperature=0.5,
            max_tokens=500
)
        
        print("summary_response",summary_response)

        summary = summary_response.choices[0].message.content
        
        print("summary",summary)

        # ナレッジ ID を取得
        knowledge_ids = [match["metadata"]["knowledge_id"] for match in response.get("matches", [])]

        # ナレッジの詳細情報を取得
        knowledges = get_knowledge_details(knowledge_ids)

        return {
            "summary": summary,
            "knowledges": knowledges
        }

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=str(e))
