from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr
import requests
import json
from db_control import crud, mymodels_MySQL
from typing import Annotated
import datetime

# MySQLのテーブル作成
from db_control.create_tables_MySQL import init_db

# アプリケーション初期化時にテーブルを作成
init_db()

# スキーマ定義
class User(BaseModel):
    id: int | None = None
    name: Annotated[str, constr(min_length = 1)]
    affiliation: Annotated[str, constr(min_length = 1)]
    email: Annotated[str, constr(min_length = 1)]
    password_hash: Annotated[str, constr(min_length = 1)]
    total_thanks: int = 0
    total_view: int = 0
    created_at: datetime.datetime | None = None

#class Meeting(BaseModel):
#    id: int | None = None
#    user_id: int
#    title: Annotated[str, constr(min_length = 1)]
#    summary: Annotated[str, constr(min_length = 1)]
#    time: datetime.time
#    created_at: datetime.datetime | None = None
    
class Knowledge(BaseModel):
    id: int | None = None
    user_id: int
    meeting_id: int
    title: Annotated[str, constr(min_length = 1)]
    content: Annotated[str, constr(min_length = 1)]
    thanks_count: int = 0
    created_at: datetime.datetime | None = None

class Challenge(BaseModel):
    id: int | None = None
    user_id: int
    meeting_id: int
    title: Annotated[str, constr(min_length = 1)]
    content: Annotated[str, constr(min_length = 1)]
    created_at: datetime.datetime | None = None

class Tag(BaseModel):
    id: int | None = None
    name: Annotated[str, constr(min_length = 1)]

class Thanks(BaseModel):
    id: int | None = None
    user_id: int
    knowledge_id: int
    created_at: datetime.datetime | None = None

class View(BaseModel):
    id: int | None = None
    user_id: int
    knowledge_id: int
    created_at: datetime.datetime | None = None

app = FastAPI()

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["https://tech0-gen-9-step3-1-node-13.azurewebsites.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {"message": "FastAPI top page!"}

@app.get("/allUser")
def get_all_User():
    result = crud.mysellectall(mymodels_MySQL.User)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)

@app.get("/allMeeting")
def get_all_Meeting():
    result = crud.mysellectall(mymodels_MySQL.Meeting)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)

@app.get("/allKnowledge")
def get_all_Knowledge():
    result = crud.mysellectall(mymodels_MySQL.Knowledge)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)

@app.get("/allChallenge")
def get_all_Challenge():
    result = crud.mysellectall(mymodels_MySQL.Challenge)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)

@app.get("/allTag")
def get_all_Tag():
    result = crud.mysellectall(mymodels_MySQL.Tag)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)

@app.get("/allKnowledgeTag")
def get_all_KnowledgeTag():
    result = crud.mysellectall(mymodels_MySQL.KnowledgeTag)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)

@app.get("/allChallengeTag")
def get_all_ChallengeTag():
    result = crud.mysellectall(mymodels_MySQL.ChallengeTag)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)

@app.get("/allThanks")
def get_all_Thanks():
    result = crud.mysellectall(mymodels_MySQL.Thanks)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)

@app.get("/allView")
def get_all_View():
    result = crud.mysellectall(mymodels_MySQL.View)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)


   
# 以下、Practicalのコード

# スキーマ定義
# class Customer(BaseModel):
#     #最低一文字を必要とする制約を設定
#     customer_id: Annotated[str, constr(min_length = 1)]
#     customer_name: Annotated[str, constr(min_length = 1)]
#     age: int
#     gender: str

# @app.post("/customers")
# def create_customer(customer: Customer):
#     values = customer.model_dump()

#     #Customer ID、顧客名が空欄の場合のエラーハンドリング
#     if values.get("customer_id") is None or values.get("customer_name") is None:
#         raise HTTPException(status_code=400, detail="This information is required!")

#     tmp = crud.myinsert(mymodels_MySQL.Customers, values)
#     result = crud.myselect(mymodels_MySQL.Customers, values.get("customer_id"))

#     if result:
#         result_obj = json.loads(result)
#         return result_obj if result_obj else None
#     return None


# @app.get("/customers")
# def read_one_customer(customer_id: str = Query(...)):
#     result = crud.myselect(mymodels_MySQL.Customers, customer_id)
#     if not result:
#         raise HTTPException(status_code=404, detail="Customer not found")
#     result_obj = json.loads(result)
#     return result_obj[0] if result_obj else None


# @app.get("/allcustomers")
# def read_all_customer():
#     result = crud.myselectAll(mymodels_MySQL.Customers)
#     # 結果がNoneの場合は空配列を返す
#     if not result:
#         return []
#     # JSON文字列をPythonオブジェクトに変換
#     return json.loads(result)


# @app.put("/customers")
# def update_customer(customer: Customer):
#     values = customer.model_dump()

#     #Customer ID、顧客名が空欄の場合のエラーハンドリング
#     if values.get("customer_id") is None or values.get("customer_name") is None:
#         raise HTTPException(status_code=400, detail="This information is required!")

#     values_original = values.copy()
#     tmp = crud.myupdate(mymodels_MySQL.Customers, values)
#     result = crud.myselect(mymodels_MySQL.Customers, values_original.get("customer_id"))
#     if not result:
#         raise HTTPException(status_code=404, detail="Customer not found")
#     result_obj = json.loads(result)
#     return result_obj[0] if result_obj else None


# @app.delete("/customers")
# def delete_customer(customer_id: str = Query(...)):
#     result = crud.mydelete(mymodels_MySQL.Customers, customer_id)
#     if not result:
#         raise HTTPException(status_code=404, detail="Customer not found")
#     return {"customer_id": customer_id, "status": "deleted"}


# @app.get("/fetchtest")
# def fetchtest():
#     response = requests.get('https://jsonplaceholder.typicode.com/users')
#     return response.json()
