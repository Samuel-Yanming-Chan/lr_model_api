
import pandas as pd


# Server
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Modeling
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Initialize files
df_train = pd.read_csv('../app/data/prep_train.csv')
df_label = df_train.loc[:, ['Survived']].copy()
df_feat = df_train.loc[:, 'Survived':].copy()

# Train model
logreg = LogisticRegression()
logreg.fit(df_feat, df_label)


class Data(BaseModel):
    Survived: int
    Pclass: int
    Sex: int
    Age: int
    SibSp: int
    Parch: int
    Fare: int
    Embarked: int
    Deck: int


@app.post("/predict")
def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        to_predict = pd.DataFrame.from_dict(data_dict)
        result = to_predict.loc[:, ['Survived']]
        test_feat = df_train.loc[:, 'Survived':]

        pred_result = logreg.predict(test_feat)
        return {"prediction": pred_result, "survived": result}

    except:
        return {"prediction": "error"}
