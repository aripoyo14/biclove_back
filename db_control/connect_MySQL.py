from sqlalchemy import create_engine
import os
from pathlib import Path
from dotenv import load_dotenv
import pymysql

# 環境変数の読み込み
base_path = Path(__file__).parents[1]
load_dotenv()

# データベース接続情報
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', "3306")
DB_NAME = os.getenv('DB_NAME')

# SSL証明書のパス
SSL_CA_PATH = str(base_path / "DigiCertGlobalRootCA.crt.pem")

# MySQLのURL構築
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# エンジンの作成
engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={
        "ssl": {
            "ssl_ca": SSL_CA_PATH
        }
    }
)
