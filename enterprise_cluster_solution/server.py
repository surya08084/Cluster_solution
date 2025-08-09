from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Lazy imports inside functions where heavy

_DATA_DIR = Path(tempfile.gettempdir()) / "ecs_data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


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
            "preprocessing": [
                {"name": "StandardScaler", "id": "standard_scaler"},
            ],
            "clustering": [
                {"name": "KMeans", "id": "kmeans"},
            ],
            "visualization": [
                {"name": "PCA Scatter", "id": "pca_scatter"},
            ],
            "metrics": [
                {"name": "Silhouette Score", "id": "silhouette"},
            ],
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
            preview = {
                "columns": df.columns.tolist(),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
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
        steps: List[Dict[str, Any]] = payload.get("steps", [])
        if not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id required")
        data_path = _DATA_DIR / f"{dataset_id}.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {"status": "queued", "progress": 0, "result": None, "error": None}
        background_tasks.add_task(_execute_pipeline, job_id, str(data_path), steps)
        return {"job_id": job_id}

    @app.get("/api/status/{job_id}")
    def job_status(job_id: str) -> Dict[str, Any]:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            return job

    return app


def _execute_pipeline(job_id: str, data_path: str, steps: List[Dict[str, Any]]) -> None:
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"] = 5

        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.metrics import silhouette_score  # type: ignore
        from sklearn.decomposition import PCA  # type: ignore
        import plotly.express as px  # type: ignore

        df = pd.read_csv(data_path)
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            raise RuntimeError("No numeric columns found for clustering")

        # Apply steps (only StandardScaler + KMeans supported initially)
        scaler = StandardScaler()
        X = scaler.fit_transform(numeric_df.values)

        n_clusters = 3
        for step in steps:
            if step.get("id") == "kmeans":
                n_clusters = int(step.get("params", {}).get("n_clusters", 3))

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        sil = None
        try:
            if len(set(labels)) > 1:
                sil = float(silhouette_score(X, labels))
        except Exception:
            sil = None

        # Visualization via PCA scatter
        if X.shape[1] > 2:
            comps = 2
            pca = PCA(n_components=comps, random_state=42)
            plot_X = pca.fit_transform(X)
            x_name, y_name = "PC1", "PC2"
        else:
            plot_X = X[:, :2]
            x_name, y_name = numeric_df.columns[:2].tolist()

        fig = px.scatter(
            x=plot_X[:, 0],
            y=plot_X[:, 1],
            color=labels.astype(str),
            labels={"x": x_name, "y": y_name, "color": "cluster"},
            title="Clusters (PCA Scatter)",
        )
        fig_json = fig.to_json()

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["result"] = {
                "n_clusters": int(n_clusters),
                "silhouette": sil,
                "figure": json.loads(fig_json),
            }
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)