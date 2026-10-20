"""
ML Algorithm Simulator & MLOps Studio - Python FastAPI Microservice
Provides real scikit-learn training, ColumnTransformer preprocessing, and joblib artifact persistence.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import io
import time
import hashlib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, silhouette_score

app = FastAPI(title="ML Algorithm Simulator - scikit-learn Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory artifact registry
MODEL_REGISTRY: Dict[str, Any] = {}

class GenerateRequest(BaseModel):
    task_type: str # 'regression' | 'classification' | 'clustering'
    samples: int = 150
    features: int = 4
    noise: float = 0.2
    seed: int = 42
    n_classes: int = 3
    n_clusters: int = 3

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "backend": "Python FastAPI + scikit-learn",
        "scikit_learn_version": "1.4.0",
        "timestamp": time.time()
    }

@app.post("/api/datasets/generate")
def generate_dataset(req: GenerateRequest):
    if req.task_type == "regression":
        X, y = make_regression(n_samples=req.samples, n_features=req.features, noise=req.noise * 10, random_state=req.seed)
        cols = [f"feature_{i+1}" for i in range(req.features)]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y
        return {"columns": list(df.columns), "data": df.to_dict(orient="records"), "task": "regression"}
    elif req.task_type == "clustering":
        X, _ = make_blobs(n_samples=req.samples, n_features=req.features, centers=req.n_clusters, random_state=req.seed)
        cols = [f"feature_{i+1}" for i in range(req.features)]
        df = pd.DataFrame(X, columns=cols)
        return {"columns": list(df.columns), "data": df.to_dict(orient="records"), "task": "clustering"}
    else:
        X, y = make_classification(n_samples=req.samples, n_features=req.features, n_classes=req.n_classes, n_informative=max(2, req.features - 1), random_state=req.seed)
        cols = [f"feature_{i+1}" for i in range(req.features)]
        df = pd.DataFrame(X, columns=cols)
        df["label"] = [f"Class_{chr(65 + c)}" for c in y]
        return {"columns": list(df.columns), "data": df.to_dict(orient="records"), "task": "classification"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
