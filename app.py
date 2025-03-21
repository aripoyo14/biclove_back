from fastapi import FastAPI
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, constr
import requests
import json
from db_control import crud, mymodels_MySQL
from typing import Annotated

# MySQLのテーブル作成
from db_control.create_tables_MySQL import init_db

# アプリケーション初期化時にテーブルを作成
init_db()

# スキーマ定義
class Customer(BaseModel):
    #最低一文字を必要とする制約を設定
    customer_id: Annotated[str, constr(min_length = 1)]
    customer_name: Annotated[str, constr(min_length = 1)]
    age: int
    gender: str

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


@app.post("/customers")
def create_customer(customer: Customer):
    values = customer.model_dump()

    #Customer ID、顧客名が空欄の場合のエラーハンドリング
    if values.get("customer_id") is None or values.get("customer_name") is None:
        raise HTTPException(status_code=400, detail="This information is required!")

    tmp = crud.myinsert(mymodels_MySQL.Customers, values)
    result = crud.myselect(mymodels_MySQL.Customers, values.get("customer_id"))

    if result:
        result_obj = json.loads(result)
        return result_obj if result_obj else None
    return None


@app.get("/customers")
def read_one_customer(customer_id: str = Query(...)):
    result = crud.myselect(mymodels_MySQL.Customers, customer_id)
    if not result:
        raise HTTPException(status_code=404, detail="Customer not found")
    result_obj = json.loads(result)
    return result_obj[0] if result_obj else None


@app.get("/allcustomers")
def read_all_customer():
    result = crud.myselectAll(mymodels_MySQL.Customers)
    # 結果がNoneの場合は空配列を返す
    if not result:
        return []
    # JSON文字列をPythonオブジェクトに変換
    return json.loads(result)


@app.put("/customers")
def update_customer(customer: Customer):
    values = customer.model_dump()

    #Customer ID、顧客名が空欄の場合のエラーハンドリング
    if values.get("customer_id") is None or values.get("customer_name") is None:
        raise HTTPException(status_code=400, detail="This information is required!")

    values_original = values.copy()
    tmp = crud.myupdate(mymodels_MySQL.Customers, values)
    result = crud.myselect(mymodels_MySQL.Customers, values_original.get("customer_id"))
    if not result:
        raise HTTPException(status_code=404, detail="Customer not found")
    result_obj = json.loads(result)
    return result_obj[0] if result_obj else None


@app.delete("/customers")
def delete_customer(customer_id: str = Query(...)):
    result = crud.mydelete(mymodels_MySQL.Customers, customer_id)
    if not result:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"customer_id": customer_id, "status": "deleted"}


@app.get("/fetchtest")
def fetchtest():
    response = requests.get('https://jsonplaceholder.typicode.com/users')
    return response.json()
