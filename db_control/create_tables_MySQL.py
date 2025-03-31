from db_control.mymodels_MySQL import Base
from db_control.connect_MySQL import engine
from sqlalchemy import inspect


def init_db():
    # インスペクターを作成
    inspector = inspect(engine)

    # 既存のテーブルを取得
    existing_tables = inspector.get_table_names()

    print("Checking tables...")

    # 必要なテーブルのリスト
    required_tables = [
        'users',
        'meeting',
        'knowledge',
        'challenge',
        'tags',
        'knowledge_tags',
        'challenge_tags',
        'thanks',
        'view'
    ]

    # 存在しないテーブルを確認
    missing_tables = [table for table in required_tables if table not in existing_tables]

    if missing_tables:
        print(f"Creating missing tables: {', '.join(missing_tables)}")
        try:
            Base.metadata.create_all(bind=engine)
            print("Tables created successfully!")
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise
    else:
        print("All required tables already exist.")


if __name__ == "__main__":
    init_db()
