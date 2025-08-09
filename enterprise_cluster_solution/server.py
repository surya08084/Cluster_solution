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
    "kmedoids": {"label": "K-Medoids", "params": {"n_clusters": 3}},
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
SUPPORTED_VISUALIZATION = ["pca", "tsne", "umap"]


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
                "imputation": {"numeric": SUPPORTED_IMPUTATION_NUMERIC, "categorical": SUPPORTED_IMPUTATION_CATEG, "per_column": True},
                "scaling": SUPPORTED_SCALING,
                "encoding": SUPPORTED_ENCODING,
                "dim_reduction": SUPPORTED_DIMRED,
                "features": {"include": True, "exclude": True},
                "outliers": {"methods": ["none", "isoforest"], "default": {"method": "isoforest", "contamination": "auto"}},
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
            import numpy as np  # type: ignore
            from io import StringIO

            df = pd.read_csv(StringIO(content.decode("utf-8")))
            suggested_dtypes = _suggest_dtypes(df)
            missing_counts = df.isna().sum().to_dict()
            missing_pct = {c: (float(m) / max(1, int(df.shape[0])) * 100.0) for c, m in missing_counts.items()}
            nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
            # Correlation preview for up to 20 numeric columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:20]
            corr_preview = None
            if len(num_cols) >= 2:
                corr = df[num_cols].corr(numeric_only=True).round(3).fillna(0.0)
                corr_preview = {"columns": num_cols, "matrix": corr.values.tolist()}
            preview = {
                "columns": df.columns.tolist(),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "suggested_dtypes": suggested_dtypes,
                "nunique": nunique,
                "missing_counts": missing_counts,
                "missing_percent": missing_pct,
                "correlation": corr_preview,
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
            "random_state": int(payload.get("random_state", 42)),
        }

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {"status": "queued", "progress": 0, "result": None, "error": None}
        background_tasks.add_task(_execute_pipeline, job_id, str(data_path), config)
        return {"job_id": job_id}

    @app.post("/api/auto_run")
    async def auto_run(payload: Dict[str, Any], background_tasks: BackgroundTasks) -> Dict[str, Any]:
        dataset_id: Optional[str] = payload.get("dataset_id")
        if not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id required")
        data_path = _DATA_DIR / f"{dataset_id}.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")

        # sensible defaults for preprocessing
        config = {
            "dtype_overrides": payload.get("dtype_overrides", {}),
            "preprocessing": payload.get(
                "preprocessing",
                {
                    "imputation": {"numeric": "mean", "categorical": "most_frequent", "fill_value": 0},
                    "encoding": "onehot",
                    "features": {},
                    "scaling": "standard",
                    "dim_reduction": {"method": "none"},
                    "outliers": {"method": "isoforest", "contamination": "auto"},
                },
            ),
            "auto": True,
            "random_state": int(payload.get("random_state", 42)),
        }

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {"status": "queued", "progress": 0, "result": None, "error": None}
        background_tasks.add_task(_execute_auto, job_id, str(data_path), config)
        return {"job_id": job_id}

    @app.post("/api/diagnostics")
    async def diagnostics(payload: Dict[str, Any]) -> Dict[str, Any]:
        dataset_id: Optional[str] = payload.get("dataset_id")
        if not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id required")
        data_path = _DATA_DIR / f"{dataset_id}.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.metrics import silhouette_score  # type: ignore

        df = pd.read_csv(data_path)
        X, _, _ = _build_features(df, payload.get("dtype_overrides", {}), payload.get("preprocessing", {}))
        # Outlier handling if provided
        out_cfg = (payload.get("preprocessing", {}) or {}).get("outliers", {"method": "none"})
        inlier_mask, _ = _handle_outliers(X, out_cfg)
        X = X[np.array(inlier_mask)]

        ks = payload.get("k_values", list(range(2, 11)))
        sil_scores = []
        inertia_vals = []
        best_k = None
        best_sil = -1.0
        for k in ks:
            try:
                km = KMeans(n_clusters=int(k), n_init=10, random_state=42)
                labels = km.fit_predict(X)
                if len(set(labels)) < 2:
                    sil = float("nan")
                else:
                    sil = float(silhouette_score(X, labels))
                sil_scores.append(sil)
                inertia_vals.append(float(km.inertia_))
                if sil == sil and sil > best_sil:  # check not nan
                    best_sil = sil
                    best_k = int(k)
            except Exception:
                sil_scores.append(float("nan"))
                inertia_vals.append(float("nan"))
        return {"k_values": ks, "silhouette": sil_scores, "inertia": inertia_vals, "suggested_k": best_k}

    @app.get("/api/status/{job_id}")
    def job_status(job_id: str) -> Dict[str, Any]:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            return job

    @app.get("/api/download/{job_id}/{name}")
    def download(job_id: str, name: str):
        job_dir = _DATA_DIR / job_id
        if name not in {"segmented.csv", "outliers.csv"}:
            raise HTTPException(status_code=400, detail="Unknown artifact")
        target = job_dir / name
        if not target.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(path=str(target), filename=name)

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


def _apply_feature_filters(df, features_cfg: Dict[str, Any]) -> Any:
    include = (features_cfg or {}).get("include")
    exclude = (features_cfg or {}).get("exclude")
    cols = list(df.columns)
    if include:
        cols = [c for c in cols if c in include]
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return df[cols]


def _build_features(
    df, dtype_overrides: Dict[str, str], preprocessing: Dict[str, Any]
) -> Tuple[Any, List[str], Dict[str, Any]]:
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

    # Feature filters
    df = _apply_feature_filters(df, (preprocessing or {}).get("features", {}))

    missing_summary = df.isna().sum().to_dict()

    # Split columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c])) and (not pd.api.types.is_datetime64_any_dtype(df[c]))]

    # Imputation
    impute_cfg = preprocessing.get("imputation", {})
    per_column = (impute_cfg or {}).get("per_column", {})
    num_strategy = impute_cfg.get("numeric", "mean")
    cat_strategy = impute_cfg.get("categorical", "most_frequent")
    fill_value = impute_cfg.get("fill_value", 0)

    X_numeric = None
    num_feature_names: List[str] = []
    if numeric_cols:
        # Per-column numeric imputation support
        if per_column:
            cols_arrays = []
            for c in numeric_cols:
                strat = per_column.get(c, {}).get("strategy", num_strategy)
                fv = per_column.get(c, {}).get("fill_value", fill_value)
                imp = SimpleImputer(strategy=strat if strat != "constant" else "constant", fill_value=fv)
                cols_arrays.append(imp.fit_transform(df[[c]]))
            X_numeric = np.hstack(cols_arrays)
        else:
            num_imputer = SimpleImputer(strategy=num_strategy if num_strategy != "constant" else "constant", fill_value=fill_value)
            X_numeric = num_imputer.fit_transform(df[numeric_cols])
        num_feature_names = numeric_cols

    X_categ = None
    cat_feature_names: List[str] = []
    if categorical_cols:
        # Per-column categorical imputation is equivalent to same strategy here
        cat_imputer = SimpleImputer(strategy=cat_strategy if cat_strategy != "constant" else "constant", fill_value=fill_value)
        cat_values = cat_imputer.fit_transform(df[categorical_cols].astype("string"))
        encoding = preprocessing.get("encoding", "onehot")
        if encoding == "onehot":
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            X_categ = encoder.fit_transform(cat_values)
            try:
                cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
            except Exception:
                cat_feature_names = [f"{categorical_cols[i]}_{j}" for i in range(len(categorical_cols)) for j in range(int(X_categ.shape[1]))]
        elif encoding == "label":
            enc_arrays = []
            for i, col in enumerate(categorical_cols):
                le = LabelEncoder()
                enc_arrays.append(le.fit_transform(cat_values[:, i]))
            import numpy as _np  # type: ignore
            X_categ = _np.vstack(enc_arrays).T
            cat_feature_names = categorical_cols
        else:
            X_categ = cat_values
            cat_feature_names = categorical_cols

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

    return X, names, {"missing_summary": missing_summary}


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

    if algo == "kmedoids":
        from sklearn_extra.cluster import KMedoids  # type: ignore

        n_clusters = int(params.get("n_clusters", 3))
        model = KMedoids(n_clusters=n_clusters, random_state=42, init="k-medoids++")
        labels = model.fit_predict(X)
        return labels.tolist(), {"n_clusters": int(n_clusters)}

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
    elif method == "umap":
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=n_components, random_state=42)
        coords = reducer.fit_transform(X)
        x_name, y_name = "UMAP1", "UMAP2"
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


def _profile_clusters(X, df, labels: List[int]) -> Dict[str, Any]:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    result: Dict[str, Any] = {}
    series_labels = pd.Series(labels, name="cluster")
    df_work = df.copy()
    df_work["cluster"] = series_labels.values

    for cluster_id, group in df_work.groupby("cluster"):
        profile: Dict[str, Any] = {}
        profile["size"] = int(group.shape[0])
        numeric_cols = group.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            desc = group[numeric_cols].describe().to_dict()
            profile["numeric_summary"] = desc
        profile["examples"] = group.head(5).drop(columns=["cluster"]).to_dict(orient="records")
        result[str(cluster_id)] = profile
    return result


def _rank_top_features(X, labels: List[int], feature_names: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
    try:
        import numpy as np  # type: ignore
        from sklearn.feature_selection import f_classif  # type: ignore

        f_stats, pvals = f_classif(X, labels)
        order = np.argsort(-f_stats)
        top = []
        for idx in order[: top_n]:
            name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
            top.append({"feature": name, "f_stat": float(f_stats[idx]), "p_value": float(pvals[idx])})
        return top
    except Exception:
        return []


def _handle_outliers(X, method_cfg: Dict[str, Any]) -> Tuple[List[bool], Dict[str, Any]]:
    method = (method_cfg or {}).get("method", "none")
    if method in (None, "none"):
        return [True] * len(X), {"method": "none", "inliers": len(X), "outliers": 0}
    if method == "isoforest":
        from sklearn.ensemble import IsolationForest  # type: ignore
        contamination = method_cfg.get("contamination", "auto")
        model = IsolationForest(random_state=42, contamination=contamination)
        preds = model.fit_predict(X)  # 1 inlier, -1 outlier
        inlier_mask = [p == 1 for p in preds]
        return inlier_mask, {"method": "isoforest", "inliers": int(sum(inlier_mask)), "outliers": int(len(inlier_mask) - sum(inlier_mask))}
    return [True] * len(X), {"method": "none", "inliers": len(X), "outliers": 0}


def _save_artifacts(job_id: str, df_inliers, labels_inliers: List[int], df_outliers) -> Dict[str, Any]:
    job_dir = _DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd  # type: ignore

    seg = df_inliers.copy()
    seg["cluster"] = labels_inliers
    seg_path = job_dir / "segmented.csv"
    seg.to_csv(seg_path, index=False)

    out_path = job_dir / "outliers.csv"
    if df_outliers is not None and len(df_outliers) > 0:
        df_outliers.to_csv(out_path, index=False)
        outliers_present = True
    else:
        outliers_present = False
    return {"segmented": str(seg_path), "outliers": (str(out_path) if outliers_present else None)}


def _execute_pipeline(job_id: str, data_path: str, config: Dict[str, Any]) -> None:
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"] = 5

        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.metrics import silhouette_score, silhouette_samples  # type: ignore

        df = pd.read_csv(data_path)

        X, feature_names, extras = _build_features(df, config.get("dtype_overrides", {}), config.get("preprocessing", {}))
        missing_summary = extras.get("missing_summary", {})

        out_cfg = (config.get("preprocessing", {}) or {}).get("outliers", {"method": "none"})
        inlier_mask, out_info = _handle_outliers(X, out_cfg)
        inlier_idx = np.where(np.array(inlier_mask))[0]
        outlier_idx = np.where(~np.array(inlier_mask))[0]

        X_in = X[inlier_idx]
        df_in = df.iloc[inlier_idx].reset_index(drop=True)
        df_out = df.iloc[outlier_idx].reset_index(drop=True)

        with _jobs_lock:
            _jobs[job_id]["progress"] = 40

        labels, algo_info = _cluster(X_in, config.get("clustering", {}))

        with _jobs_lock:
            _jobs[job_id]["progress"] = 70

        sil = None
        sil_samples = None
        try:
            unique = set(labels)
            if len(unique) >= 2:
                sil = float(silhouette_score(X_in, labels))
                sil_samples = silhouette_samples(X_in, labels).tolist()
        except Exception:
            sil = None
            sil_samples = None

        fig_json = _visualize(X_in, labels, config.get("visualization", {"method": "pca", "n_components": 2}))

        from collections import Counter

        counts = Counter(labels)
        profiles = _profile_clusters(X_in, df_in, labels)
        top_features = _rank_top_features(X_in, labels, feature_names)
        artifacts = _save_artifacts(job_id, df_in, labels, df_out)

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["result"] = {
                "labels": labels,
                "label_counts": dict(counts),
                "silhouette": sil,
                "silhouette_samples": sil_samples,
                "algorithm": config.get("clustering", {}).get("algorithm", "kmeans"),
                "algorithm_params": algo_info,
                "figure": fig_json,
                "profiles": profiles,
                "top_features": top_features,
                "missing_summary": missing_summary,
                "outlier_summary": out_info,
                "artifacts": {"segmented": bool(artifacts.get("segmented")), "outliers": bool(artifacts.get("outliers"))},
            }
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)


def _execute_auto(job_id: str, data_path: str, config: Dict[str, Any]) -> None:
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"] = 5

        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.metrics import silhouette_score  # type: ignore

        df = pd.read_csv(data_path)

        X, feature_names, extras = _build_features(df, config.get("dtype_overrides", {}), config.get("preprocessing", {}))
        missing_summary = extras.get("missing_summary", {})

        out_cfg = (config.get("preprocessing", {}) or {}).get("outliers", {"method": "isoforest", "contamination": "auto"})
        inlier_mask, out_info = _handle_outliers(X, out_cfg)
        inlier_idx = np.where(np.array(inlier_mask))[0]
        outlier_idx = np.where(~np.array(inlier_mask))[0]

        X_in = X[inlier_idx]
        df_in = df.iloc[inlier_idx].reset_index(drop=True)
        df_out = df.iloc[outlier_idx].reset_index(drop=True)

        candidate_algos = [
            ("kmeans", {"n_clusters": k}) for k in (2, 3, 4, 5, 6)
        ] + [
            ("kmedoids", {"n_clusters": k}) for k in (2, 3, 4, 5)
        ] + [
            ("agglomerative", {"n_clusters": k, "linkage": "ward"}) for k in (2, 3, 4, 5)
        ] + [
            ("gmm", {"n_components": k}) for k in (2, 3, 4, 5)
        ] + [
            ("dbscan", {"eps": e, "min_samples": 5}) for e in (0.3, 0.5, 0.8)
        ]

        best = {"score": -1.0, "algo": None, "params": None, "labels": None}

        for algo, params in candidate_algos:
            try:
                labels, _ = _cluster(X_in, {"algorithm": algo, "params": params})
                unique = set(labels)
                if len(unique) < 2:
                    continue
                if algo == "dbscan" and labels.count(-1) > 0.5 * len(labels):
                    continue
                if -1 in unique and len(unique) > 2:
                    mask = np.array(labels) != -1
                    score = float(silhouette_score(X_in[mask], np.array(labels)[mask]))
                else:
                    score = float(silhouette_score(X_in, labels))
                if score > best["score"]:
                    best = {"score": score, "algo": algo, "params": params, "labels": labels}
            except Exception:
                continue

        if best["algo"] is None:
            raise RuntimeError("Auto run failed to find a valid clustering")

        vis = _visualize(X_in, best["labels"], {"method": "umap", "n_components": 2})
        from collections import Counter

        counts = Counter(best["labels"])  # type: ignore
        profiles = _profile_clusters(X_in, df_in, best["labels"])  # type: ignore
        top_features = _rank_top_features(X_in, best["labels"], feature_names)  # type: ignore
        artifacts = _save_artifacts(job_id, df_in, best["labels"], df_out)

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 100
            _jobs[job_id]["result"] = {
                "labels": best["labels"],
                "label_counts": dict(counts),
                "silhouette": best["score"],
                "algorithm": best["algo"],
                "algorithm_params": best["params"],
                "figure": vis,
                "profiles": profiles,
                "top_features": top_features,
                "missing_summary": missing_summary,
                "outlier_summary": out_info,
                "artifacts": {"segmented": bool(artifacts.get("segmented")), "outliers": bool(artifacts.get("outliers"))},
            }
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)