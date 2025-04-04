# biclove_back

## 起動方法

```
# Powershell を管理者権限で起動し、以下の SQL 立ち上げのコマンドを実行
net start mysql57
mysql --user=root --password
exit

# リモートリポジトリからコードをクローンしてローカルで作業する準備
git clone https://github.com/aripoyo14/biclove_back.git
cd biclove_back
python -m venv venv
pip install -r requirements.txt

touch .env
# .env ファイルを書き換える（内容についてはこれから調べます）

uvicorn app:app --reload
```
