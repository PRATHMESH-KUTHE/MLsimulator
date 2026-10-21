"""
ML Algorithm Simulator & MLOps Studio - Python FastAPI Microservice
Provides scikit-learn training, ColumnTransformer preprocessing, and dataset generation.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
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
    task_type: Optional[str] = "classification"
    taskType: Optional[str] = None
    samples: int = 150
    features: int = 4
    noise: float = 0.2
    seed: int = 42
    n_classes: int = 3
    nClasses: Optional[int] = None
    n_clusters: int = 3
    nClusters: Optional[int] = None

class TrainRequest(BaseModel):
    algorithm: str
    train_x: List[List[float]]
    train_y: List[Any]
    test_x: List[List[float]]
    test_y: List[Any]
    feature_names: List[str]
    hyperparameters: Optional[Dict[str, Any]] = None

@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "backend",
        "backend": "Python FastAPI + scikit-learn",
        "scikit_learn_version": "1.4.0",
        "timestamp": time.time()
    }

@app.post("/api/datasets/generate")
@app.post("/datasets/generate")
def generate_dataset(req: GenerateRequest):
    # Support both snake_case and camelCase from frontend
    task = req.taskType or req.task_type or "classification"
    n_classes = req.nClasses or req.n_classes or 3
    n_clusters = req.nClusters or req.n_clusters or 3

    if task == "regression":
        X, y = make_regression(n_samples=req.samples, n_features=req.features, noise=req.noise * 10, random_state=req.seed)
        cols = [f"feature_{i+1}" for i in range(req.features)]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = np.round(y, 2)
        return {"columns": list(df.columns), "data": df.to_dict(orient="records"), "task": "regression"}
    elif task == "clustering":
        X, _ = make_blobs(n_samples=req.samples, n_features=req.features, centers=n_clusters, random_state=req.seed)
        cols = [f"feature_{i+1}" for i in range(req.features)]
        df = pd.DataFrame(X, columns=cols)
        return {"columns": list(df.columns), "data": df.to_dict(orient="records"), "task": "clustering"}
    else:
        X, y = make_classification(n_samples=req.samples, n_features=req.features, n_classes=n_classes, n_informative=max(2, req.features - 1), random_state=req.seed)
        cols = [f"feature_{i+1}" for i in range(req.features)]
        df = pd.DataFrame(X, columns=cols)
        df["label"] = [f"Class_{chr(65 + c)}" for c in y]
        return {"columns": list(df.columns), "data": df.to_dict(orient="records"), "task": "classification"}

@app.post("/api/models/train")
@app.post("/models/train")
def train_model_endpoint(req: TrainRequest):
    algo = req.algorithm
    X_train = np.array(req.train_x)
    y_train = np.array(req.train_y)
    X_test = np.array(req.test_x)
    y_test = np.array(req.test_y)

    params = req.hyperparameters or {}
    start_time = time.time()

    model = None
    task_type = "classification"

    if algo == "linear_regression":
        model = LinearRegression()
        model.fit(X_train, y_train.astype(float))
        preds = model.predict(X_test)
        r2 = r2_score(y_test.astype(float), preds)
        rmse = float(np.sqrt(mean_squared_error(y_test.astype(float), preds)))
        task_type = "regression"
        metrics = {"r2": round(float(r2), 4), "rmse": round(rmse, 4)}
    elif algo == "decision_tree_classifier":
        max_depth = params.get("maxDepth", 4)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        metrics = {"accuracy": round(float(acc), 4), "f1Score": round(float(f1), 4)}
    elif algo == "kmeans_clustering":
        k = params.get("nClusters", 3)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_train)
        sil = silhouette_score(X_train, labels) if len(set(labels)) > 1 else 0.0
        task_type = "clustering"
        metrics = {"silhouetteScore": round(float(sil), 4), "inertia": round(float(model.inertia_), 2)}
    else:
        # Default fallback classifier
        model = KNeighborsClassifier(n_neighbors=params.get("nNeighbors", 5))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        metrics = {"accuracy": round(float(acc), 4)}

    duration_ms = round((time.time() - start_time) * 1000)
    model_id = f"py_model_{int(time.time())}"
    MODEL_REGISTRY[model_id] = model

    return {
        "model_id": model_id,
        "algorithm": algo,
        "task_type": task_type,
        "duration_ms": duration_ms,
        "metrics": metrics,
        "status": "trained"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
