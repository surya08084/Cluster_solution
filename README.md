# Enterprise Cluster Solution

A Python package that launches a local web UI to build and run clustering pipelines with minimal code.

## Installation

```bash
pip install -e .
```

This installs in editable mode for development. For production, build a wheel and install it.

## Quickstart

```python
from enterprise_cluster_solution import ClusteringSolution

model = ClusteringSolution()
url = model.launch_ui()
print("UI available at:", url)
```

Open the URL in your browser to upload data, configure a simple pipeline (StandardScaler + KMeans), and view interactive results.

## Features (initial subset)
- Drag-and-drop placeholder UI for pipeline ordering
- CSV upload and data preview
- Preprocessing: Standardization
- Clustering: KMeans
- Metrics: Silhouette score
- Visualization: 2D scatter via PCA with cluster coloring (if >2 features)

## Roadmap
- Additional preprocessing modules, clustering algorithms, evaluation metrics
- Branching pipelines and job monitoring
- Authentication, project save/load, and artifact downloads

## License
MIT