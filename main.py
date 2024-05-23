from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional


class TextDto(BaseModel):
    Left: float
    Top: float
    Right: float
    Bottom: float
    Color: str
    Family: Optional[str] = None
    Size: str
    LineSpace: str
    
    class Config:
        extra = "allow"  # Allow additional fields


class requestDto(BaseModel):
    textPos: List[TextDto]
    outputNum: Optional[int] = 10


class IdDto(BaseModel):
    Predicted_TemplateId: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("init lifespan")
    app.model = joblib.load("./knn_model.pkl")
    app.sca = joblib.load("./scaler.pkl")
    app.y = joblib.load("./y.pkl")
    app.max_len = 8
    yield
    app.model.clear()
    app.sca.clear()
    app.y.clear()
    # Clean up the ML models and release the resources
    print("clean up lifespan")


app = FastAPI(lifespan=lifespan)

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/getTemplate")
async def postAi(request: requestDto) -> IdDto:
    test_feature = []
    textPos = request.textPos
    outputNum = request.outputNum

    for text in textPos:
        test_feature.extend([text.Top, text.Right, text.Bottom, text.Left, text.Size])
    if len(test_feature) < app.max_len * 5:
        test_feature.extend([0] * (app.max_len * 5 - len(test_feature)))
    elif len(test_feature) > app.max_len * 5:
        test_feature = test_feature[: app.max_len * 5]
    train_vector = app.sca.transform([test_feature])
    # predicted = app.model.predict(train_vector)
    distances, indices = app.model.kneighbors(train_vector, n_neighbors=outputNum)
    
    ids = []
    for i in indices:
        templateId = app.y.iloc[i].values.tolist()
        ids.extend(templateId)
    return IdDto(Predicted_TemplateId=ids)
