from __future__ import annotations

from typing import Any, Dict, Optional


def autoaicluster(
    data: Any = None,
    *,
    ui: bool = False,
    dtype_overrides: Optional[Dict[str, str]] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
    candidate_k: Optional[list[int]] = None,
    open_browser: bool = True,
) -> Any:
    """
    One-call entry point.
    - If ui=True or data is None: launch the local web UI and return the URL.
    - Else: run AutoML clustering on the provided pandas DataFrame or CSV file path and return a result dict.
    """
    if ui or data is None:
        from enterprise_cluster_solution import ClusteringSolution

        model = ClusteringSolution()
        return model.launch_ui(open_browser=open_browser)

    from enterprise_cluster_solution import auto_segment

    return auto_segment(
        data=data,
        dtype_overrides=dtype_overrides,
        preprocessing=preprocessing,
        candidate_k=candidate_k,
    )