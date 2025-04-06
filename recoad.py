# FastAPI関連のインポート
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr
from typing import Annotated, List, Optional

# データベース関連のインポート
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from db_control import mymodels_MySQL, crud
from db_control.connect_MySQL import engine

# OpenAI関連のインポート
from openai import OpenAI  # 音声認識用の直接のOpenAIクライアント
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

# スキーマ関連のインポート
from schemas import MeetingResponse, SolutionKnowledgeRequest, SolutionKnowledgeResponse

# SessionLocal を定義（connect_MySQL.py の engine を利用）
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# .env ファイルの読み込み（環境変数を使うため）
load_dotenv()

# OpenAI APIキーを環境変数から取得して設定
api_key = os.getenv("OPENAI_API_KEY")

# Pineconeの設定を環境変数から取得
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

# 環境変数が正しく設定されているかチェック
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment variables.")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 環境変数のバリデーション
if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
    raise RuntimeError("Pinecone configuration is missing in environment variables")

# Pineconeクライアントの初期化
pc = Pinecone(api_key=PINECONE_API_KEY)

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

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


#DBチェック
for key in ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]:
    if not os.getenv(key):
        raise RuntimeError(f"Missing database config: {key}")


# FastAPIアプリの作成
app = FastAPI()

# CORS設定（フロントエンドと通信を可能にする）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # セキュリティを考慮して本番環境では制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title generation error: {e}")

# GPT出力のパース関数（要約 / 知見 / 悩み に分ける）
def parse_summary_response(summary_text: str) -> dict:
    try:
        summary_match = re.search(r"①【会議の要約】\n?(.*?)(?:\n②|$)", summary_text, re.DOTALL)
        knowledge_match = re.search(r"②【知見】\n?(.*?)(?:\n③|$)", summary_text, re.DOTALL)
        challenge_match = re.search(r"③【悩み】\n?(.*)", summary_text, re.DOTALL)

        summary = summary_match.group(1).strip() if summary_match else ""
        knowledge = [line.lstrip("-・ ").strip() for line in knowledge_match.group(1).split("\n") if line.strip()] if knowledge_match else []
        challenge = [line.lstrip("-・ ").strip() for line in challenge_match.group(1).split("\n") if line.strip()] if challenge_match else []

        return {
            "summary": summary,
            "knowledges": knowledge,
            "challenges": challenge,
        }
    except Exception as e:
        raise ValueError(f"GPT出力のパースに失敗しました: {e}")

# タグ名とテキストが部分一致するかチェック
def find_matching_tags(text: str, tags: list) -> list:
    return [tag.name for tag in tags if tag.name in text]

# ルートエンドポイント（APIが稼働しているか確認する用）
@app.get("/")
async def root():
    return {"message": "Recoad API is running independently!"}

# 音声ファイルを受け取り、文字起こしし、データベースに保存するエンドポイント
@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    user_id: int = Form(...) # ← フロントから受け取る！
):
    try:
        # 音声ファイルをサーバーに保存
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 文字起こし処理
        transcript = await transcribe_audio(file_path)
        
        # 要約処理
        summary_text = generate_summary(transcript)
        parsed = parse_summary_response(summary_text) #パースで分ける
        title = generate_title(parsed["summary"])
        
        # データベースに保存
        db: Session = SessionLocal()
        try:
            all_tags = db.query(mymodels_MySQL.Tag).all()
            new_meeting = mymodels_MySQL.Meeting(
                user_id=user_id,
                title=title,
                summary=parsed["summary"],#パースで分ける
                #time=datetime.time(hour=0, minute=30)  # 仮時間
            )
            db.add(new_meeting)
            db.commit()
            db.refresh(new_meeting)
            
            meeting_id = new_meeting.id  # ← セッションを閉じる前にIDを保持！
            
            #知見登録
            for knowledge_content in parsed["knowledges"]: #パースで分ける
                knowledge = mymodels_MySQL.Knowledge(
                    user_id=user_id,
                    meeting_id=meeting_id,
                    title="自動生成知見タイトル",
                    content=knowledge_content
                )
                matched_tags = find_matching_tags(knowledge_content, all_tags)
                for tag_name in matched_tags:
                    tag = db.query(mymodels_MySQL.Tag).filter_by(name=tag_name).first()
                    if tag and tag not in knowledge.tags:
                        knowledge.tags.append(tag)
                db.add(knowledge)
                
            # 悩み登録
            for challenge_content in parsed["challenges"]: #パースで分ける
                challenge = mymodels_MySQL.Challenge(
                    user_id=user_id,
                    meeting_id=meeting_id,
                    title="自動生成課題タイトル",
                    content=challenge_content
                )
                matched_tags = find_matching_tags(challenge_content, all_tags)
                for tag_name in matched_tags:
                    tag = db.query(mymodels_MySQL.Tag).filter_by(name=tag_name).first()
                    if tag and tag not in challenge.tags:
                        challenge.tags.append(tag)
                db.add(challenge)
                
            db.commit()
                
        finally:
            db.close()

        response_data = {
            "message": "Meeting + 知見 + 悩み 登録完了",
            "meeting_id": meeting_id,
            "title": title,
            "transcript": transcript,
            "parsed_summary": {
                "summary": parsed["summary"],
                "knowledges": [
                    {
                        "id": k.id,
                        "content": k.content
                    }
                    for k in db.query(mymodels_MySQL.Knowledge).filter_by(meeting_id=meeting_id).all()
                ],
                "challenges": [
                    {
                        "id": c.id,
                        "content": c.content
                    }
                    for c in db.query(mymodels_MySQL.Challenge).filter_by(meeting_id=meeting_id).all()
                ]
            }
        }

        # 🔒 DBセッションを閉じる
        db.close()

        # 📤 セッション終了後にreturnする！
        return response_data

            
    except Exception as e:
        traceback.print_exc()  # ← ★追加
        raise HTTPException(status_code=500, detail=str(e))

# 会議の文字起こしデータを取得し、要約を生成するエンドポイント(ORM版)
@app.get("/get-summary/{meeting_id}")
async def get_summary(meeting_id: int):
    db: Session = SessionLocal()
    try:
        meeting = db.query(mymodels_MySQL.Meeting).filter(mymodels_MySQL.Meeting.id == meeting_id).first()
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")

        if meeting.summary:
            return {"summary": meeting.summary}

        # 文字起こしデータを要約
        summary = generate_summary(meeting.transcript)

        # DBに要約を保存
        meeting.summary = summary
        db.commit()
        return {"summary": summary}

    finally:
        db.close()
        
@app.get("/get-knowledge/{meeting_id}")
async def get_knowledge(meeting_id: int):
    db: Session = SessionLocal()
    try:
        knowledges = db.query(mymodels_MySQL.Knowledge).filter(mymodels_MySQL.Knowledge.meeting_id == meeting_id).all()
        return knowledges

        # 文字起こしデータを要約
        #summary = generate_summary(meeting.transcript)

        # DBに要約を保存
        #meeting.summary = summary
        db.commit()
        return {"result": "error"}

    finally:
        db.close()
        
# フロント側で編集された会議情報を更新するエンドポイント
@app.put("/update-meeting/{meeting_id}")
async def update_meeting(meeting_id: int, data: dict = Body(...)):
    db: Session = SessionLocal()
    try:
        meeting = db.query(mymodels_MySQL.Meeting).filter_by(id=meeting_id).first()
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")

        # タイトル・サマリを更新
        meeting.title = data.get("title", meeting.title)
        meeting.summary = data.get("summary", meeting.summary)

        # タグ一覧取得（名前で紐付け）
        all_tags = db.query(mymodels_MySQL.Tag).all()

        # 知見の更新
        for k in data.get("knowledges", []):
            knowledge = db.query(mymodels_MySQL.Knowledge).filter_by(id=k["id"]).first()
            if knowledge:
                knowledge.title = k["title"]
                knowledge.content = k["content"]
                knowledge.tags = []  # 一旦タグを空にする
                for tag_name in k.get("tags", []):
                    tag = next((t for t in all_tags if t.name == tag_name), None)
                    if tag:
                        knowledge.tags.append(tag)
                        
        # 課題の更新
        for c in data.get("challenges", []):
            challenge = db.query(mymodels_MySQL.Challenge).filter_by(id=c["id"]).first()
            if challenge:
                challenge.title = c["title"]
                challenge.content = c["content"]
                challenge.tags = []  # 一旦タグを空にする
                for tag_name in c.get("tags", []):
                    tag = next((t for t in all_tags if t.name == tag_name), None)
                    if tag:
                        challenge.tags.append(tag)

        db.commit()
        
        challenges = db.query(mymodels_MySQL.Challenge).filter(mymodels_MySQL.Challenge.meeting_id == meeting_id).all()
        knowledges = db.query(mymodels_MySQL.Knowledge).filter(mymodels_MySQL.Knowledge.meeting_id == meeting_id).all()

        # Challeneges と Knowledges を一つの文書にまとめる
        all_content = "## 課題\n-"+ "\n- ".join([challenge.content for challenge in challenges]) + "\n\n## 知見\n-"+ "\n- ".join([knowledge.content for knowledge in knowledges])
        create_index(all_content)
        
        return {"message": "Meeting内容を更新しベクトル化しました"}

    finally:
        db.close()

def create_index(content: str):
    """
    ナレッジの内容をベクトル化してPineconeに保存する
    
    Args:
        content: ベクトル化する内容
    """
    try:
        # ベクトル化
        knowledge_vector = embeddings.embed_query(content)
        
        # ユニークなIDを生成（タイムスタンプとハッシュを使用）
        unique_id = hashlib.md5(content.encode()).hexdigest()
        vector_id = f"vec_{int(time.time())}_{unique_id[:10]}"

        # Pineconeへの保存
        index.upsert([
            (
                vector_id, 
                knowledge_vector,
                {
                    "content": content,
                    "created_at": datetime.datetime.now().isoformat()
                }
            )
        ])
        
        return {"status": "success", "vector_id": vector_id}
        
    except Exception as e:
        print("❌️ ", str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "ベクトル化処理に失敗しました"}
        )

@app.get("/vectors")
async def get_all_vectors():
    try:
        # まずインデックスの統計を取得
        stats = index.describe_index_stats()
        total_vectors = stats['total_vector_count']
        
        # インデックス内のすべてのベクトルを取得
        fetch_response = index.query(
            vector=[0] * 1536,  # ダミーのクエリベクトル
            top_k=total_vectors,
            include_metadata=True
        )
        
        # レスポンスデータを整形（循環参照を避ける）
        vectors = []
        for match in fetch_response['matches']:
            vector_data = {
                "id": match.id,
                "score": float(match.score) if match.score else None,  # numpy.float64をPythonのfloatに変換
                "metadata": match.metadata
            }
            vectors.append(vector_data)
        
        return {
            "total_vectors": total_vectors,
            "vectors": vectors
        }
    except Exception as e:
        error_detail = {
            "message": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        raise HTTPException(
            status_code=500,
            detail={"error": "ベクトル取得エラー", "details": error_detail}
        )

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
        print(response)
        knowledge_texts = [match["metadata"]["content"] for match in response.get("matches", [])]
        combined_knowledge = "\n".join(knowledge_texts)
        print(combined_knowledge)

        # GPT による要約生成
        summary_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "以下のユーザーの質問に対する回答をコンテキスト情報を元に500文字以内で生成してください。コンテキスト情報に含まれない内容は回答しないでください"},
                {"role": "user", "content": f"""
ユーザーの質問：
{challenge.content}

コンテキスト：
{combined_knowledge}
"""
        }
            ],
            temperature=0.5,
            max_tokens=500
        )

        summary = summary_response.choices[0].message.content

        # ナレッジ ID を取得
        knowledge_ids = [match["id"] for match in response.get("matches", [])]
        # ナレッジの詳細情報を取得
        # knowledges = crud.get_knowledge_details(knowledge_ids)

        # print("✅ knowledges")
        # print(knowledges)   

        return {
            "summary": summary,
            "knowledges": knowledge_ids
        }

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=str(e))