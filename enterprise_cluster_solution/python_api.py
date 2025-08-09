from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union


def _as_dataframe(data: Any):
    import pandas as pd  # type: ignore

    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, str):
        return pd.read_csv(data)
    raise TypeError("data must be a pandas.DataFrame or a CSV filepath string")


def run_pipeline(
    data: Any,
    dtype_overrides: Optional[Dict[str, str]] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
    clustering: Optional[Dict[str, Any]] = None,
    visualization: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a single clustering pipeline on a DataFrame or CSV path.

    Returns a dict with keys: labels, label_counts, silhouette, algorithm, algorithm_params,
    figure (plotly JSON), profiles, top_features, missing_summary, outlier_summary.
    """
    dtype_overrides = dtype_overrides or {}
    preprocessing = preprocessing or {}
    clustering = clustering or {"algorithm": "kmeans", "params": {"n_clusters": 3}}
    visualization = visualization or {"method": "pca", "n_components": 2}

    import numpy as np  # type: ignore
    from collections import Counter

    from .server import (
        _build_features,
        _handle_outliers,
        _cluster,
        _visualize,
        _profile_clusters,
        _rank_top_features,
    )

    df = _as_dataframe(data)

    X, feature_names, extras = _build_features(df, dtype_overrides, preprocessing)
    missing_summary = extras.get("missing_summary", {})

    out_cfg = (preprocessing or {}).get("outliers", {"method": "none"})
    inlier_mask, out_info = _handle_outliers(X, out_cfg)

    inlier_idx = np.where(np.array(inlier_mask))[0]
    X_in = X[inlier_idx]
    df_in = df.iloc[inlier_idx].reset_index(drop=True)

    labels, algo_info = _cluster(X_in, clustering)

    # silhouette
    sil = None
    try:
        from sklearn.metrics import silhouette_score  # type: ignore

        unique = set(labels)
        if len(unique) >= 2:
            sil = float(silhouette_score(X_in, labels))
    except Exception:
        sil = None

    fig_json = _visualize(X_in, labels, visualization)

    counts = Counter(labels)
    profiles = _profile_clusters(X_in, df_in, labels)
    top_features = _rank_top_features(X_in, labels, feature_names)

    return {
        "labels": labels,
        "label_counts": dict(counts),
        "silhouette": sil,
        "algorithm": clustering.get("algorithm", "kmeans"),
        "algorithm_params": algo_info,
        "figure": fig_json,
        "profiles": profiles,
        "top_features": top_features,
        "missing_summary": missing_summary,
        "outlier_summary": out_info,
    }


def auto_segment(
    data: Any,
    dtype_overrides: Optional[Dict[str, str]] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
    candidate_k: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    AutoML-like segmentation: tries several algorithms/params and selects the best silhouette.

    Returns a dict like run_pipeline with the winning algorithm and labels.
    """
    dtype_overrides = dtype_overrides or {}
    preprocessing = preprocessing or {
        "imputation": {"numeric": "mean", "categorical": "most_frequent", "fill_value": 0},
        "encoding": "onehot",
        "scaling": "standard",
        "dim_reduction": {"method": "none"},
        "outliers": {"method": "isoforest", "contamination": "auto"},
    }

    import numpy as np  # type: ignore
    from collections import Counter

    from .server import (
        _build_features,
        _handle_outliers,
        _cluster,
        _visualize,
        _profile_clusters,
        _rank_top_features,
    )

    from sklearn.metrics import silhouette_score  # type: ignore

    df = _as_dataframe(data)

    X, feature_names, extras = _build_features(df, dtype_overrides, preprocessing)
    missing_summary = extras.get("missing_summary", {})

    out_cfg = (preprocessing or {}).get("outliers", {"method": "isoforest", "contamination": "auto"})
    inlier_mask, out_info = _handle_outliers(X, out_cfg)

    inlier_idx = np.where(np.array(inlier_mask))[0]
    X_in = X[inlier_idx]
    df_in = df.iloc[inlier_idx].reset_index(drop=True)

    # candidate settings
    candidate_algos: List[Tuple[str, Dict[str, Any]]] = []
    k_values = candidate_k or [2, 3, 4, 5, 6]
    for k in k_values:
        candidate_algos.append(("kmeans", {"n_clusters": k}))
        if k <= 5:
            candidate_algos.append(("kmedoids", {"n_clusters": k}))
            candidate_algos.append(("agglomerative", {"n_clusters": k, "linkage": "ward"}))
            candidate_algos.append(("gmm", {"n_components": k}))
    for e in (0.3, 0.5, 0.8):
        candidate_algos.append(("dbscan", {"eps": e, "min_samples": 5}))

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
        raise RuntimeError("Auto segment failed to find a valid clustering")

    vis = _visualize(X_in, best["labels"], {"method": "umap", "n_components": 2})

    counts = Counter(best["labels"])  # type: ignore
    profiles = _profile_clusters(X_in, df_in, best["labels"])  # type: ignore
    top_features = _rank_top_features(X_in, best["labels"], feature_names)  # type: ignore

    return {
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
    }