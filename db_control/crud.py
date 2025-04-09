# uname() error回避
import platform
print("platform", platform.uname())


from sqlalchemy import create_engine, insert, delete, update, select
import sqlalchemy
from sqlalchemy.orm import sessionmaker, Session
import json
import pandas as pd
from typing import List
import hashlib
import time
from fastapi import HTTPException
import traceback

from db_control.connect_MySQL import engine
from db_control.mymodels_MySQL import User, Meeting, Knowledge, Challenge, Thanks, View

def mysellectall(mymodel):
    # session構築
    Session = sessionmaker(bind=engine)
    session = Session()
    query = select(mymodel)
    try:
        # トランザクションを開始
        with session.begin():
            df = pd.read_sql_query(query, con=engine)
            result_json = df.to_json(orient='records', force_ascii=False)

    except sqlalchemy.exc.IntegrityError:
        print("一意制約違反により、挿入に失敗しました")
        result_json = None

    # セッションを閉じる
    session.close()
    return result_json

# -----------------------------------------------------------------------------
# get_meeting_with_related_data_using_join_optimized
# 最新の会議データと関連するチャレンジとナレッジを取得する最適化された関数
# フロントのサイドバーに4件の会議タイトルが表示されるので4件の会議データを取得する
# -----------------------------------------------------------------------------
def get_meeting_with_related_data_using_join_optimized(user_id: int = None, limit=4):
    """
    JOINを利用して一度のクエリで最新の会議データと関連するチャレンジとナレッジを取得する最適化された関数
    
    Args:
        user_id (int, optional): ユーザーID（指定された場合、そのユーザーの会議のみを取得）
        limit (int): 取得する会議の数（デフォルト: 4）
        
    Returns:
        list: 会議データと関連するチャレンジとナレッジのリスト
    """
    # session構築
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 最新の会議データを取得（IDが最大のものから指定数）
        meetings_query = select(Meeting)
        if user_id is not None:
            meetings_query = meetings_query.where(Meeting.user_id == user_id)
        meetings_query = meetings_query.order_by(Meeting.id.desc()).limit(limit)
        meetings_df = pd.read_sql_query(meetings_query, con=engine)
        meetings_data = json.loads(meetings_df.to_json(orient='records', force_ascii=False))
        
        # 会議IDのリストを作成
        meeting_ids = [meeting['id'] for meeting in meetings_data]
        
        # 関連するチャレンジとナレッジを一度に取得
        from sqlalchemy import union_all
        
        # チャレンジとナレッジを結合するクエリ
        challenges_query = select(
            Challenge.id.label('id'),
            Challenge.user_id.label('user_id'),
            Challenge.meeting_id.label('meeting_id'),
            Challenge.title.label('title'),
            Challenge.content.label('content'),
            Challenge.created_at.label('created_at'),
            sqlalchemy.literal('challenge').label('type')
        ).where(Challenge.meeting_id.in_(meeting_ids))
        
        knowledges_query = select(
            Knowledge.id.label('id'),
            Knowledge.user_id.label('user_id'),
            Knowledge.meeting_id.label('meeting_id'),
            Knowledge.title.label('title'),
            Knowledge.content.label('content'),
            Knowledge.created_at.label('created_at'),
            sqlalchemy.literal('knowledge').label('type')
        ).where(Knowledge.meeting_id.in_(meeting_ids))
        
        # クエリを結合
        combined_query = union_all(challenges_query, knowledges_query)
        
        # クエリを実行
        combined_df = pd.read_sql_query(combined_query, con=engine)
        combined_data = json.loads(combined_df.to_json(orient='records', force_ascii=False))
        
        # データをマージ
        result = []
        for meeting in meetings_data:
            meeting_id = meeting['id']
            
            # 関連するチャレンジをフィルタリング
            meeting_challenges = [item for item in combined_data if item['meeting_id'] == meeting_id and item['type'] == 'challenge']
            
            # 関連するナレッジをフィルタリング
            meeting_knowledges = [item for item in combined_data if item['meeting_id'] == meeting_id and item['type'] == 'knowledge']
            
            # データをマージ
            meeting['challenges'] = meeting_challenges
            meeting['knowledges'] = meeting_knowledges
            result.append(meeting)
        
        return result
    except sqlalchemy.exc.IntegrityError:
        print("一意制約違反により、取得に失敗しました")
        return []
    finally:
        session.close()

def get_knowledge_details(knowledge_ids: List[int]):
    """
    指定されたナレッジIDの詳細情報を取得する関数
    
    Args:
        knowledge_ids (List[int]): ナレッジIDのリスト
        
    Returns:
        list: ナレッジの詳細情報のリスト（ユーザー名を含む）
    """
    # session構築
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # ナレッジとユーザーの情報を結合して取得
        query = select(
            Knowledge.id,
            Knowledge.title,
            Knowledge.content,
            Knowledge.user_id,
            User.name.label('user_name')
        ).join(
            User, Knowledge.user_id == User.id
        ).where(Knowledge.id.in_(knowledge_ids))
        
        df = pd.read_sql_query(query, con=engine)
        result = json.loads(df.to_json(orient='records', force_ascii=False))
        
        return result
    except sqlalchemy.exc.IntegrityError:
        print("一意制約違反により、取得に失敗しました")
        return []
    finally:
        session.close()

def create_index(knowledges: list, index, embeddings):
    """
    指定された knowledges リストの内容をベクトル化し、Pinecone 等のベクトルDBに保存する。
    
    Args:
        knowledges: ベクトル化する知見のリスト [{"id": int, "content": str}, ...]
        index: Pineconeのインデックスオブジェクト
        embeddings: ベクトル化用のモデル
    """
    try:
        # ベクトルDBへの保存用データ
        vectors_to_upsert = []
        
        for knowledge in knowledges:
            # ベクトル化処理
            knowledge_vector = embeddings.embed_query(knowledge["content"])
            
            # ユニークなベクトルIDの生成（タイムスタンプとコンテンツのハッシュを利用）
            unique_id = hashlib.md5(knowledge["content"].encode()).hexdigest()
            vector_id = f"vec_{int(time.time())}_{unique_id[:10]}"
            
            # アップサート用のタプルを作成
            vectors_to_upsert.append((
                vector_id, 
                knowledge_vector,
                {
                    "knowledge_id": knowledge["id"],
                    "text": knowledge["content"]
                }
            ))
        
        # もしアップサート対象があれば、ベクトルDBに一括登録
        if vectors_to_upsert:
            index.upsert(vectors_to_upsert)
        
        return {"status": "success", "count": len(vectors_to_upsert)}
        
    except Exception as e:
        print("❌️ ", str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "ベクトル化処理に失敗しました"}
        )

def get_all_vectors(index):
    """
    ベクトルデータベースからすべてのベクトルを取得する関数
    
    Args:
        index: Pineconeのインデックスオブジェクト
        
    Returns:
        dict: ベクトルデータの統計情報とベクトルリスト
    """
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
                "score": float(match.score) if match.score else None,
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

def save_meeting_with_knowledge_and_challenge(
    db: Session,
    user_id: int,
    title: str,
    summary: str,
    knowledges: list,
    challenges: list
):
    """
    会議データと関連する知見、課題をデータベースに保存する関数
    
    Args:
        db (Session): データベースセッション
        user_id (int): ユーザーID
        title (str): 会議のタイトル
        summary (str): 会議の要約
        knowledges (list): 知見のリスト [{"title": str, "content": str}, ...]
        challenges (list): 課題のリスト [{"title": str, "content": str}, ...]
        
    Returns:
        dict: 保存したデータの情報
    """
    try:
        # 会議の保存
        new_meeting = Meeting(
            user_id=user_id,
            title=title,
            summary=summary
        )
        db.add(new_meeting)
        db.commit()
        db.refresh(new_meeting)
        meeting_id = new_meeting.id

        # 知見の保存
        knowledge_list = []
        for knowledge_item in knowledges:
            knowledge = Knowledge(
                user_id=user_id,
                meeting_id=meeting_id,
                title=knowledge_item["title"],
                content=knowledge_item["content"]
            )
            db.add(knowledge)
            db.flush()
            db.refresh(knowledge)
            knowledge_list.append(knowledge)

        # 課題の保存
        challenge_list = []
        for challenge_item in challenges:
            challenge = Challenge(
                user_id=user_id,
                meeting_id=meeting_id,
                title=challenge_item["title"],
                content=challenge_item["content"]
            )
            db.add(challenge)
            db.flush()
            db.refresh(challenge)
            challenge_list.append(challenge)

        db.commit()

        return {
            "meeting_id": meeting_id,
            "meeting": {
                "id": meeting_id,
                "title": title,
                "summary": summary,
            },
            "knowledges": [
                {"id": k.id, "title": k.title, "content": k.content}
                for k in knowledge_list
            ],
            "challenges": [
                {"id": c.id, "title": c.title, "content": c.content}
                for c in challenge_list
            ]
        }
    except Exception as e:
        db.rollback()
        raise e

def get_meeting_summary(db: Session, meeting_id: int):
    """
    指定された会議IDの要約を取得する関数
    
    Args:
        db (Session): データベースセッション
        meeting_id (int): 会議ID
        
    Returns:
        dict: 会議の要約情報
    """
    try:
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")

        if meeting.summary:
            return {"summary": meeting.summary}

        return {"summary": None}
    except Exception as e:
        raise e

def get_meeting_knowledges(db: Session, meeting_id: int):
    """
    指定された会議IDの知見を取得する関数
    
    Args:
        db (Session): データベースセッション
        meeting_id (int): 会議ID
        
    Returns:
        list: 知見のリスト
    """
    try:
        knowledges = db.query(Knowledge).filter(Knowledge.meeting_id == meeting_id).all()
        return knowledges
    except Exception as e:
        raise e