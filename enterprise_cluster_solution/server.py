from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Lazy imports inside functions where heavy

_DATA_DIR = Path(tempfile.gettempdir()) / "ecs_data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


SUPPORTED_ALGOS = {
    "kmeans": {"label": "KMeans", "params": {"n_clusters": 3}},
    "agglomerative": {"label": "Agglomerative", "params": {"n_clusters": 3, "linkage": "ward"}},
    "birch": {"label": "BIRCH", "params": {"n_clusters": 3}},
    "dbscan": {"label": "DBSCAN", "params": {"eps": 0.5, "min_samples": 5}},
    "optics": {"label": "OPTICS", "params": {"min_samples": 5}},
    "gmm": {"label": "Gaussian Mixture", "params": {"n_components": 3}},
}

SUPPORTED_IMPUTATION_NUMERIC = ["mean", "median", "constant"]
SUPPORTED_IMPUTATION_CATEG = ["most_frequent", "constant"]
SUPPORTED_SCALING = ["none", "standard", "minmax", "robust"]
SUPPORTED_ENCODING = ["none", "onehot", "label"]
SUPPORTED_DIMRED = ["none", "pca"]
SUPPORTED_VISUALIZATION = ["pca", "tsne"]


def build_app() -> FastAPI:
    app = FastAPI(title="Enterprise Cluster Solution", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent / "ui"
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        index_file = static_dir / "index.html"
        if not index_file.exists():
            return HTMLResponse(content="<h1>UI not found</h1>", status_code=500)
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

    @app.get("/api/modules")
    def list_modules() -> Dict[str, Any]:
        return {
            "preprocessing": {
                "imputation": {"numeric": SUPPORTED_IMPUTATION_NUMERIC, "categorical": SUPPORTED_IMPUTATION_CATEG},
                "scaling": SUPPORTED_SCALING,
                "encoding": SUPPORTED_ENCODING,
                "dim_reduction": SUPPORTED_DIMRED,
            },
            "clustering": SUPPORTED_ALGOS,
            "visualization": SUPPORTED_VISUALIZATION,
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".csv"}:  # initial subset
            raise HTTPException(status_code=400, detail="Only CSV supported in this build")
        dataset_id = str(uuid.uuid4())
        dest = _DATA_DIR / f"{dataset_id}.csv"
        content = await file.read()
        dest.write_bytes(content)
        # basic preview using pandas
        try:
            import pandas as pd  # type: ignore
            from io import StringIO

            df = pd.read_csv(StringIO(content.decode("utf-8")))
            suggested_dtypes = _suggest_dtypes(df)
            preview = {
                "columns": df.columns.tolist(),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "suggested_dtypes": suggested_dtypes,
                "head": df.head(5).to_dict(orient="records"),
                "num_rows": int(df.shape[0]),
                "num_columns": int(df.shape[1]),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}")
        return {"dataset_id": dataset_id, "preview": preview}

    @app.post("/api/run")
    async def run_pipeline(payload: Dict[str, Any], background_tasks: BackgroundTasks) -> Dict[str, Any]:
        dataset_id: Optional[str] = payload.get("dataset_id")
        if not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id required")
        data_path = _DATA_DIR / f"{dataset_id}.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        config = {
            "dtype_overrides": payload.get("dtype_overrides", {}),
            "preprocessing": payload.get("preprocessing", {}),
            "clustering": payload.get("clustering", {}),
            "visualization": payload.get("visualization", {"method": "pca", "n_components": 2}),
        }

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {"status": "queued", "progress": 0, "result": None, "error": None}
        background_tasks.add_task(_execute_pipeline, job_id, str(data_path), config)
        return {"job_id": job_id}

    @app.get("/api/status/{job_id}")
    def job_status(job_id: str) -> Dict[str, Any]:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            return job

    return app


def _suggest_dtypes(df) -> Dict[str, str]:
    import numpy as np  # type: ignore

    suggestions: Dict[str, str] = {}
    for col in df.columns:
        dt = df[col].dtype
        if np.issubdtype(dt, np.number):
            suggestions[col] = "numeric"
        else:
            suggestions[col] = "categorical"
    return suggestions


def _build_features(
    df, dtype_overrides: Dict[str, str], preprocessing: Dict[str, Any]
) -> Tuple[Any, List[str]]:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler  # type: ignore

    # Apply dtype overrides
    for col, kind in dtype_overrides.items():
        if col not in df.columns:
            continue
        if kind == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif kind == "categorical":
            df[col] = df[col].astype("string")
        elif kind == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Split columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c])) and (not pd.api.types.is_datetime64_any_dtype(df[c]))]

    # Imputation
    impute_cfg = preprocessing.get("imputation", {})
    num_strategy = impute_cfg.get("numeric", "mean")
    cat_strategy = impute_cfg.get("categorical", "most_frequent")
    fill_value = impute_cfg.get("fill_value", 0)

    X_numeric = None
    num_feature_names: List[str] = []
    if numeric_cols:
        num_imputer = SimpleImputer(strategy=num_strategy if num_strategy != "constant" else "constant", fill_value=fill_value)
        X_numeric = num_imputer.fit_transform(df[numeric_cols])
        num_feature_names = numeric_cols

    X_categ = None
    cat_feature_names: List[str] = []
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=cat_strategy if cat_strategy != "constant" else "constant", fill_value=fill_value)
        cat_values = cat_imputer.fit_transform(df[categorical_cols].astype("string"))
        encoding = preprocessing.get("encoding", "onehot")
        if encoding == "onehot":
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            X_categ = encoder.fit_transform(cat_values)
            # Build names
            try:
                cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
            except Exception:
                cat_feature_names = [f"{categorical_cols[i]}_{j}" for i in range(len(categorical_cols)) for j in range(int(X_categ.shape[1]))]
        elif encoding == "label":
            # Label encode each column separately then concatenate
            enc_arrays = []
            for i, col in enumerate(categorical_cols):
                le = LabelEncoder()
                enc_arrays.append(le.fit_transform(cat_values[:, i]))
            X_categ = np.vstack(enc_arrays).T
            cat_feature_names = categorical_cols
        else:
            X_categ = cat_values
            cat_feature_names = categorical_cols

    # Concatenate features
    parts = []
    names: List[str] = []
    if X_numeric is not None:
        parts.append(X_numeric)
        names.extend(num_feature_names)
    if X_categ is not None:
        parts.append(X_categ)
        names.extend(cat_feature_names)

    if not parts:
        raise RuntimeError("No usable features after preprocessing")

    import numpy as np  # type: ignore

    X = np.concatenate(parts, axis=1)

    # Scaling
    scaling = preprocessing.get("scaling", "standard")
    if scaling == "standard":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaling == "minmax":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif scaling == "robust":
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

    # Optional pre-clustering dimensionality reduction (PCA only for now)
    dim_cfg = preprocessing.get("dim_reduction", {"method": "none"})
    if isinstance(dim_cfg, str):
        dim_method = dim_cfg
        dim_n = 2
    else:
        dim_method = dim_cfg.get("method", "none")
        dim_n = int(dim_cfg.get("n_components", 2))
    if dim_method == "pca" and X.shape[1] > dim_n:
        from sklearn.decomposition import PCA  # type: ignore
        X = PCA(n_components=dim_n, random_state=42).fit_transform(X)

    return X, names


def _cluster(X, clustering_cfg: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
    algo = clustering_cfg.get("algorithm", "kmeans")
    params = clustering_cfg.get("params", {})

    if algo == "kmeans":
        from sklearn.cluster import KMeans  # type: ignore

        n_clusters = int(params.get("n_clusters", 3))
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = model.fit_predict(X)
        info = {"n_clusters": int(n_clusters)}
        return labels.tolist(), info

    if algo == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering  # type: ignore

        n_clusters = int(params.get("n_clusters", 3))
        linkage = str(params.get("linkage", "ward"))
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X)
        return labels.tolist(), {"n_clusters": int(n_clusters), "linkage": linkage}

    if algo == "birch":
        from sklearn.cluster import Birch  # type: ignore

        n_clusters = int(params.get("n_clusters", 3))
        model = Birch(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        return labels.tolist(), {"n_clusters": int(n_clusters)}

    if algo == "dbscan":
        from sklearn.cluster import DBSCAN  # type: ignore

        eps = float(params.get("eps", 0.5))
        min_samples = int(params.get("min_samples", 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        return labels.tolist(), {"eps": eps, "min_samples": int(min_samples)}

    if algo == "optics":
        from sklearn.cluster import OPTICS  # type: ignore

        min_samples = int(params.get("min_samples", 5))
        model = OPTICS(min_samples=min_samples)
        labels = model.fit_predict(X)
        return labels.tolist(), {"min_samples": int(min_samples)}

    if algo == "gmm":
        from sklearn.mixture import GaussianMixture  # type: ignore

        n_components = int(params.get("n_components", 3))
        model = GaussianMixture(n_components=n_components, random_state=42)
        labels = model.fit_predict(X)
        return labels.tolist(), {"n_components": int(n_components)}

    raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algo}")


def _visualize(X, labels: List[int], vis_cfg: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np  # type: ignore
    import plotly.express as px  # type: ignore

    method = vis_cfg.get("method", "pca")
    n_components = int(vis_cfg.get("n_components", 2))
    n_components = 2 if n_components not in (2, 3) else n_components

    if method == "tsne":
        from sklearn.manifold import TSNE  # type: ignore

        reducer = TSNE(n_components=n_components, random_state=42, init="random", learning_rate="auto")
        coords = reducer.fit_transform(X)
        x_name, y_name = "TSNE1", "TSNE2"
    else:
        from sklearn.decomposition import PCA  # type: ignore

        reducer = PCA(n_components=min(n_components, X.shape[1]), random_state=42)
        coords = reducer.fit_transform(X)
        x_name, y_name = ("PC1", "PC2")

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=[str(c) for c in labels],
        labels={"x": x_name, "y": y_name, "color": "cluster"},
        title=f"Clusters ({method.upper()} Scatter)",
    )
    return json.loads(fig.to_json())


def _execute_pipeline(job_id: str, data_path: str, config: Dict[str, Any]) -> None:
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"] = 5

        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.metrics import silhouette_score  # type: ignore

        df = pd.read_csv(data_path)

        X, feature_names = _build_features(df, config.get("dtype_overrides", {}), config.get("preprocessing", {}))

        with _jobs_lock:
            _jobs[job_id]["progress"] = 40

        labels, algo_info = _cluster(X, config.get("clustering", {}))

        with _jobs_lock:
            _jobs[job_id]["progress"] = 70

        # Compute silhouette if valid (>=2 clusters, no single cluster)
        sil = None
        try:
            unique = set(labels)
            if len(unique) >= 2 and (len(unique) != 1):
                # Filter out noise label -1 for silhouette when present
                if -1 in unique and len(unique) > 2:
                    mask = np.array(labels) != -1
                    sil = float(silhouette_score(X[mask], np.array(labels)[mask]))
                else:
                    sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

        fig_json = _visualize(X, labels, config.get("visualization", {"method": "pca", "n_components": 2}))

        # Label counts
        from collections import Counter

        counts = Counter(labels)

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["result"] = {
                "labels": labels,
                "label_counts": dict(counts),
                "silhouette": sil,
                "algorithm": config.get("clustering", {}).get("algorithm", "kmeans"),
                "algorithm_params": algo_info,
                "figure": fig_json,
            }
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)