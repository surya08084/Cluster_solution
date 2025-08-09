# Auto AI Cluster

End-to-end clustering (segmentation) via Python API or local web UI.

## Install

```bash
pip install autoaicluster
```

## One-liner usage

```python
from autoaicluster import autoaicluster

# Launch web UI
url = autoaicluster(ui=True)
print(url)

# Or run AutoML clustering directly on a DataFrame or CSV path
import pandas as pd

df = pd.DataFrame({
    'age':[25,37,29,41,33,52,47,23,39,31],
    'income':[40,72,50,90,60,120,95,35,70,55],
    'city':['A','B','A','B','A','B','B','A','B','A']
})
res = autoaicluster(df)
print(res['algorithm'], res['silhouette'], res['label_counts'])
```

## Advanced (Python API)
- Full control via `enterprise_cluster_solution.auto_segment` and `run_pipeline` for preprocessing and algorithm selection.

## Notes
- To access UI from other machines, set env vars before launching: `ECS_HOST=0.0.0.0 ECS_PORT=8000`.

## License
MIT