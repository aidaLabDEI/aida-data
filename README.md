# AIDA-data

This repository contains code for downloading and preprocessing datasets
used across software project of the [AIDA Lab](https://aidalabdei.github.io/)
at the Department of Information Engineering of the University of Padova.

## Usage

<details>
<summary>Dense datasets.</summary>

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aida_data.dense import SafeL2Normalizer, load

# Raw (cleaned) data
distance, train = load("glove-100-angular")

# Preprocess with any sklearn transformer/pipeline: it is fitted on the
# train set and applied to the test set.
pipeline = make_pipeline(StandardScaler(with_std=False), PCA(64), SafeL2Normalizer())
ds = load("glove-100-angular", pipeline, load_queries=True)
print(pipeline.named_steps["pca"].explained_variance_ratio_)
```

</details>

<details>
<summary>Graph datasets with colored nodes.</summary>

Graphs from [Sirius](https://github.com/leonardopellegrina/Sirius/tree/main/data),
loaded as undirected simple graphs (self loops and duplicate edges dropped,
each edge stored once as `(u, v)` with `u < v`).

```python
from aida_data import graph

print(graph.available_datasets())
g = graph.load_edge_list("brexit")
g.edges   # (m, 2) int64 array
g.colors  # (n,) int64 array, g.colors[v] is the color of node v (-1 if missing)
g.n_nodes, g.n_edges, g.n_colors

# Keep edge orientation, and remap colors to 0..k-1
g = graph.load_edge_list("brexit", directed=True, remap_colors=True)
```

</details>
