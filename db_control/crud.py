# uname() error回避
import platform
print("platform", platform.uname())


from sqlalchemy import create_engine, insert, delete, update, select
import sqlalchemy
from sqlalchemy.orm import sessionmaker
import json
import pandas as pd

from db_control.connect_MySQL import engine
from db_control.mymodels_MySQL import User, Meeting, Knowledge, Challenge, Tag, KnowledgeTag, ChallengeTag, Thanks, View

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
            Challenge.meeting_id.label('meeting_id'),
            Challenge.user_id.label('user_id'),
            Challenge.title.label('title'),
            Challenge.content.label('content'),
            Challenge.created_at.label('created_at'),
            sqlalchemy.literal('challenge').label('type')
        ).where(Challenge.meeting_id.in_(meeting_ids))
        
        knowledges_query = select(
            Knowledge.id.label('id'),
            Knowledge.meeting_id.label('meeting_id'),
            Knowledge.user_id.label('user_id'),
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

# -----------------------------------------------------------------------------
# get_meeting_with_related_data_using_join_optimizedと同じ機能の為、コメントアウト
# -----------------------------------------------------------------------------
# def get_meeting_with_related_data_using_join(limit=4):
#     """
#     JOINを利用して一度のクエリで最新の会議データと関連するチャレンジとナレッジを取得する関数
    
#     Args:
#         limit (int): 取得する会議の数（デフォルト: 4）
        
#     Returns:
#         list: 会議データと関連するチャレンジとナレッジのリスト
#     """
#     # session構築
#     Session = sessionmaker(bind=engine)
#     session = Session()
    
#     try:
#         # 最新の会議データを取得（IDが最大のものから指定数）
#         meetings_query = select(Meeting).order_by(Meeting.id.desc()).limit(limit)
#         meetings_df = pd.read_sql_query(meetings_query, con=engine)
#         meetings_data = json.loads(meetings_df.to_json(orient='records', force_ascii=False))
        
#         # 会議IDのリストを作成
#         meeting_ids = [meeting['id'] for meeting in meetings_data]
        
#         # 関連するチャレンジを取得
#         challenges_query = select(Challenge).where(Challenge.meeting_id.in_(meeting_ids))
#         challenges_df = pd.read_sql_query(challenges_query, con=engine)
#         challenges_data = json.loads(challenges_df.to_json(orient='records', force_ascii=False))
        
#         # 関連するナレッジを取得
#         knowledges_query = select(Knowledge).where(Knowledge.meeting_id.in_(meeting_ids))
#         knowledges_df = pd.read_sql_query(knowledges_query, con=engine)
#         knowledges_data = json.loads(knowledges_df.to_json(orient='records', force_ascii=False))
        
#         # データをマージ
#         result = []
#         for meeting in meetings_data:
#             meeting_id = meeting['id']
            
#             # 関連するチャレンジをフィルタリング
#             meeting_challenges = [challenge for challenge in challenges_data if challenge['meeting_id'] == meeting_id]
            
#             # 関連するナレッジをフィルタリング
#             meeting_knowledges = [knowledge for knowledge in knowledges_data if knowledge['meeting_id'] == meeting_id]
            
#             # データをマージ
#             meeting['challenges'] = meeting_challenges
#             meeting['knowledges'] = meeting_knowledges
#             result.append(meeting)
        
#         return result
#     except sqlalchemy.exc.IntegrityError:
#         print("一意制約違反により、取得に失敗しました")
#         return []
#     finally:
#         session.close()

# -----------------------------------------------------------------------------
# get_meeting_with_related_data_using_join()に必要だったもの。今は不要
# -----------------------------------------------------------------------------
# def get_latest_meetings(limit=4):
#     """
#     最新の会議データを取得する関数
    
#     Args:
#         limit (int): 取得する会議の数（デフォルト: 4）
        
#     Returns:
#         str: JSON形式の会議データ
#     """
#     # session構築
#     Session = sessionmaker(bind=engine)
#     session = Session()
    
#     try:
#         # 最新の会議データを取得（IDが最大のものから指定数）
#         query = select(Meeting).order_by(Meeting.id.desc()).limit(limit)
#         df = pd.read_sql_query(query, con=engine)
#         result_json = df.to_json(orient='records', force_ascii=False)
#         return result_json
#     except sqlalchemy.exc.IntegrityError:
#         print("一意制約違反により、取得に失敗しました")
#         return None
#     finally:
#         session.close()

# def get_challenges_by_meeting_id(meeting_id):
#     """
#     特定の会議IDに関連するチャレンジを取得する関数
    
#     Args:
#         meeting_id (int): 会議ID
        
#     Returns:
#         str: JSON形式のチャレンジデータ
#     """
#     # session構築
#     Session = sessionmaker(bind=engine)
#     session = Session()
    
#     try:
#         # 特定の会議IDに関連するチャレンジを取得
#         query = select(Challenge).where(Challenge.meeting_id == meeting_id)
#         df = pd.read_sql_query(query, con=engine)
#         result_json = df.to_json(orient='records', force_ascii=False)
#         return result_json
#     except sqlalchemy.exc.IntegrityError:
#         print("一意制約違反により、取得に失敗しました")
#         return None
#     finally:
#         session.close()

# def get_knowledges_by_meeting_id(meeting_id):
#     """
#     特定の会議IDに関連するナレッジを取得する関数
    
#     Args:
#         meeting_id (int): 会議ID
        
#     Returns:
#         str: JSON形式のナレッジデータ
#     """
#     # session構築
#     Session = sessionmaker(bind=engine)
#     session = Session()
    
#     try:
#         # 特定の会議IDに関連するナレッジを取得
#         query = select(Knowledge).where(Knowledge.meeting_id == meeting_id)
#         df = pd.read_sql_query(query, con=engine)
#         result_json = df.to_json(orient='records', force_ascii=False)
#         return result_json
#     except sqlalchemy.exc.IntegrityError:
#         print("一意制約違反により、取得に失敗しました")
#         return None
#     finally:
#         session.close()

# def get_meeting_with_related_data(limit=4):
#     """
#     最新の会議データと関連するチャレンジとナレッジを取得する関数
    
#     Args:
#         limit (int): 取得する会議の数（デフォルト: 4）
        
#     Returns:
#         list: 会議データと関連するチャレンジとナレッジのリスト
#     """
#     # 最新の会議データを取得
#     meetings_json = get_latest_meetings(limit)
#     if not meetings_json:
#         return []
    
#     meetings_data = json.loads(meetings_json)
#     result = []
    
#     for meeting in meetings_data:
#         meeting_id = meeting['id']
        
#         # 関連するチャレンジを取得
#         challenges_json = get_challenges_by_meeting_id(meeting_id)
#         challenges_data = json.loads(challenges_json) if challenges_json else []
        
#         # 関連するナレッジを取得
#         knowledges_json = get_knowledges_by_meeting_id(meeting_id)
#         knowledges_data = json.loads(knowledges_json) if knowledges_json else []
        
#         # データをマージ
#         meeting['challenges'] = challenges_data
#         meeting['knowledges'] = knowledges_data
#         result.append(meeting)
    
#     return result