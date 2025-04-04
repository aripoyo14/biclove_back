from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session, sessionmaker
from db_control import mymodels_MySQL
from db_control.connect_MySQL import engine  # 既存のエンジンを利用
import datetime
import re
import traceback  # ← これを追加！


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
                file=audio_file,
                model="whisper-1"
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
@app.post("/upload-audio/")
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
                time=datetime.time(hour=0, minute=30)  # 仮時間
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
        return {"message": "Meeting内容を更新しました"}

    finally:
        db.close()

