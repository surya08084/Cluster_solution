# Enterprise Cluster Solution

A Python package that launches a local web UI to build and run clustering pipelines with minimal code.

## Installation

```bash
python3 -m pip install --break-system-packages -e .
```

## Quickstart (Web UI)

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

## Python Workflow (No UI)

```python
import pandas as pd
from enterprise_cluster_solution import run_pipeline, auto_segment

# dataframe or CSV path
df = pd.DataFrame({
    'age':[25,37,29,41,33,52,47,23,39,31],
    'income':[40,72,50,90,60,120,95,35,70,55],
    'city':['A','B','A','B','A','B','B','A','B','A']
})

# AutoML-style segmentation
result = auto_segment(df)
print('Algorithm:', result['algorithm'])
print('Silhouette:', result['silhouette'])
print('Counts:', result['label_counts'])

# Or manual pipeline
pipeline = run_pipeline(
    df,
    preprocessing={
        'imputation': {'numeric': 'mean', 'categorical': 'most_frequent', 'fill_value': 0},
        'encoding': 'onehot',
        'scaling': 'standard',
        'outliers': {'method': 'isoforest', 'contamination': 'auto'}
    },
    clustering={'algorithm': 'kmeans', 'params': {'n_clusters': 3}},
    visualization={'method': 'pca', 'n_components': 2}
)
print('KMeans silhouette:', pipeline['silhouette'])
```

## Notes on access from containers/remote
- If running UI in a container or remote VM, bind to `0.0.0.0` and open `http://<public-ip>:<port>`.

## Features
- Dtype editing; per-column and global imputation; include/exclude columns
- Outlier handling (IsolationForest), separate outliers artifact
- Algorithms: KMeans, K-Medoids, Agglomerative, BIRCH, DBSCAN, OPTICS, GMM
- Visualization: PCA, t-SNE, UMAP
- Metrics/insights: Silhouette, per-point silhouette, top features, cluster profiles
- Auto Segment: tries algorithms/params and selects best silhouette
- Artifacts: segmented.csv, outliers.csv

## License
MIT