from fastapi import FastAPI
from pydantic import BaseModel  # リクエストbodyを定義するために必要
import datetime

app = FastAPI()


# リクエストbodyを定義
class User(BaseModel):
    user_id: int
    name: str

class Meeting(BaseModel):
    id: int | None = None
    user_id: int
    title: str
    summary: str
    time: datetime.time
    created_at: datetime.datetime | None = None



# シンプルなJSON Bodyの受け取り
@app.post("/user/")
# 上で定義したUserモデルのリクエストbodyをuserで受け取る
# user = {"user_id": 1, "name": "太郎"}
def create_user(user: User):
    # レスポンスbody
    return {"res": "ok", "ID": user.user_id, "名前": user.name}


@app.post("/meeting")
def post_finalized_meeting(newMeeting: Meeting):
    print(newMeeting)
    return {"res": "ok", "ID": newMeeting.id}