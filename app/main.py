
"""
This app creates a FastAPI instance.  It will import the data and train the model when it initialization.
Make an API call to: http://127.0.0.1:8000/predict and it will return a prediction and the actual label.
"""


# Server
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Modeling
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Initialize files
df_train = pd.read_csv('../app/data/prep_train.csv')
df_label = df_train.loc[:, ['Survived']].copy()
df_feat = df_train.iloc[:, 1:].copy()

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
        to_predict = pd.DataFrame(data_dict, index=[0])
        result = data_dict['Survived']  # Save Result to compare against prediction
        test_feat = to_predict.iloc[:, 1:]

        pred_result = logreg.predict(test_feat)     # Predict result from Request
        pred_result = pred_result[0].item()
        return {"prediction": pred_result, "result": result}

    except:
        return {"prediction": "error"}
