# biclove_back

# 開発環境セットアップ手順

## 1. 必要なソフトウェアのインストール

### Python のインストール

1. [Python 公式サイト](https://www.python.org/downloads/)から Python 3.11 をダウンロード
2. インストーラーを実行（「Add Python to PATH」にチェックを入れる）
3. インストール完了後、ターミナルで確認：

```bash
python --version  # Python 3.11.x と表示されることを確認
```

### MySQL のインストール

1. [MySQL 公式サイト](https://dev.mysql.com/downloads/mysql/)から MySQL 5.7 をダウンロード
2. インストーラーを実行
   - root ユーザーのパスワードを設定（忘れないように注意）
   - 「Start MySQL Server at System Startup」にチェック
3. インストール完了後、MySQL サービスの起動：

```bash
# Windows の場合
net start mysql57

# Mac の場合
brew services start mysql@5.7
```

## 2. プロジェクトのセットアップ

### リポジトリのクローンと仮想環境の作成

```bash
# リポジトリをクローン
git clone https://github.com/aripoyo14/biclove_back.git
cd biclove_back

# 仮想環境を作成して有効化
# Windows の場合
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux の場合
python -m venv venv
source venv/bin/activate
```

### 依存パッケージのインストール

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 環境変数の設定

`.env`ファイルをプロジェクトルートに作成し、以下の内容を設定：

```env
# Database
MYSQL_USER=root
MYSQL_PASSWORD=【MySQLインストール時に設定したrootパスワード】
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=biclove

# OpenAI
OPENAI_API_KEY=【OpenAIのAPIキー】

# Pinecone
PINECONE_API_KEY=【PineconeのAPIキー】
PINECONE_ENVIRONMENT=【Pineconeの環境名】
PINECONE_INDEX_NAME=【Pineconeのインデックス名】
```

## 4. データベースの初期設定

```bash
# MySQL に接続
mysql -u root -p
# パスワードを入力

# データベースを作成
CREATE DATABASE biclove_flowledge;
exit;
```

## 5. アプリケーションの起動

```bash
# 開発サーバーを起動
uvicorn recoad:app --reload
```

## 6. アプリケーションと仮想環境の終了方法

### アプリケーションの終了

```bash
# ターミナルで Ctrl + C を押す
```

### 仮想環境の終了

```bash
deactivate
```

### MySQL サービスの停止（必要な場合）

```bash
# Windows の場合
net stop mysql57

# Mac の場合
brew services stop mysql@5.7
```

## 動作確認

ブラウザで http://127.0.0.1:8000/docs にアクセスし、Swagger UI が表示されることを確認

## 注意事項

- OpenAI と Pinecone の API キーは事前に取得しておく必要があります
- MySQL のバージョンは 5.7 を推奨します
- 環境変数の値は実際の値に置き換えてください
- エラーが発生した場合は、各種ログを確認してください

## トラブルシューティング

- `pip install`でエラーが発生する場合：
  - Windows の場合：Visual C++ Build Tools をインストール
  - Mac の場合：`xcode-select --install`を実行
- MySQL 接続エラーの場合：
  - MySQL サービスが起動していることを確認
  - 環境変数の接続情報が正しいことを確認
