# Enterprise Cluster Solution

A Python package that launches a local web UI to build and run clustering pipelines with minimal code.

## Installation

```bash
python3 -m pip install --break-system-packages -e .
```

## Quickstart

```python
from enterprise_cluster_solution import ClusteringSolution

# Binds to 0.0.0.0 by default; shows a localhost URL
model = ClusteringSolution()
url = model.launch_ui()
print("UI available at:", url)
```

- To override host/port, use environment variables or constructor:
  - `ECS_HOST` or `HOST` (e.g., 0.0.0.0 for external access)
  - `ECS_PORT` or `PORT`
  - Example: `ECS_HOST=0.0.0.0 ECS_PORT=8000 python -c 'from enterprise_cluster_solution import ClusteringSolution; print(ClusteringSolution().launch_ui(False))'`

## Notes on access from containers/remote
- If running in a container or remote VM, bind to `0.0.0.0` and open `http://<public-ip>:<port>`.
- Firewalls or port forwarding may block access; ensure the chosen port is open.

## Features (initial subset)
- Dtype editing; preprocessing (impute, encode, scale, optional PCA)
- Algorithms: KMeans, K-Medoids, Agglomerative, BIRCH, DBSCAN, OPTICS, GMM
- Visualization: PCA, t-SNE, UMAP
- Metrics: Silhouette; cluster profiles (size, numeric summary, examples)
- Auto Run: tries multiple algorithms to pick best silhouette

## Roadmap
- External metrics (ARI, NMI), branching pipelines, artifact downloads

## License
MIT