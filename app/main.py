import pandas as pd

# Server
import uvicorn
from fastapi import FastAPI

# Modeling
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Initialize files
df_train = pd.read_csv(r'../data/prep_train.csv')
df_label = df_train.loc[:, ['Survived']]
df_feat = df_train.iloc[:, 0:]

# Train model
logreg = LogisticRegression()
logreg.fit(df_feat, df_label)


class Data(BaseModel):
    Survived: bool
    Pclass: int
    Sex: bool
    Age: int
    SibSip: int
    Parch: int
    Fare: int
    Embarked: int
    Deck: int


def predict(data: Data):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        to_predict = pd.DataFrame.from_dict(data_dict)
        result = to_predict.loc[:, ['Survived']]
        test_feat = df_train.iloc[:, 0:]

        pred_result = logreg.predict(test_feat)
        return {"prediction": pred_result, "survived": result}

    except:
        return {"prediction": "error"}
