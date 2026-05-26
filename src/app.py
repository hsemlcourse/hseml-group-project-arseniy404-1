import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Codeforces Rating Predictor",
    description="Machine learning service for Codeforces problem rating prediction",
    version="1.0.0",
)

model = joblib.load("../models/best_model.pkl")


class ProblemRequest(BaseModel):
    statement: str
    solvedCount: int
    n_tags: int
    statement_length: int
    statement_words: int
    has_math: int
    name_length: int
    name_words: int

    tag_binary_search: int = 0
    tag_constructive_algorithms: int = 0
    tag_brute_force: int = 0
    tag_greedy: int = 0
    tag_bitmasks: int = 0
    tag_combinatorics: int = 0
    tag_data_structures: int = 0
    tag_dp: int = 0
    tag_math: int = 0
    tag_implementation: int = 0


@app.get("/")
def root():
    return {
        "message": "Codeforces Rating Predictor API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(request: ProblemRequest):
    log_solved = np.log1p(request.solvedCount)

    data = pd.DataFrame(
        [
            {
                "solvedCount": request.solvedCount,
                "n_tags": request.n_tags,
                "statement_length": request.statement_length,
                "statement_words": request.statement_words,
                "has_math": request.has_math,
                "name_length": request.name_length,
                "name_words": request.name_words,
                "log_solved": log_solved,

                "tag_binary search": request.tag_binary_search,
                "tag_constructive algorithms": request.tag_constructive_algorithms,
                "tag_brute force": request.tag_brute_force,
                "tag_greedy": request.tag_greedy,
                "tag_bitmasks": request.tag_bitmasks,
                "tag_combinatorics": request.tag_combinatorics,
                "tag_data structures": request.tag_data_structures,
                "tag_dp": request.tag_dp,
                "tag_math": request.tag_math,
                "tag_implementation": request.tag_implementation,
            }
        ]
    )

    prediction = model.predict(data)[0]

    return {
        "predicted_rating": int(round(prediction, -2))
    }