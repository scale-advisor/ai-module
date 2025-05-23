from typing import Union
from fastapi import FastAPI, Body
import json
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from .dataset import FPDataset
from model import load_model, load_tokenizer, create_trainer
from config import max_length, random_seed
from predict import predict_fp_type


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/classify")
def classify(input_json: List[Dict] = Body(...)):

    data = pd.DataFrame(input_json)

    # 예측할 unitProcess 목록
    data_list = data['unitProcess'].tolist()
    id_list = data['unitProcessId'].tolist()

    # 예측
    predicted_fp_types = predict_fp_type(data_list)

    # 결과 DataFrame 생성
    result_df = pd.DataFrame({
        'unitProcessId': id_list,
        'unitProcess': data_list,
        'functionType': predicted_fp_types
    })

    # 유형이 'X'인 항목 제거
    result_df = result_df[result_df['functionType'] != 'X'].reset_index(drop=True)

    return result_df.to_dict(orient="records") 
    # return result_df.to_json(orient='records', force_ascii=False, indent=4)

