## [Unreleased]

### 🚀 Features

- *(dense)* Optional row ids (`Dataset.ids`, `Dataset.query_ids`) stored in the parquet cache
- *(mteb)* Encode MTEB Retrieval and Clustering tasks into the dense parquet cache
- *(dense)* Load `mteb:<task>:<model>` names, from the local cache, the hosted registry or a local encode
- *(dense)* Add `load_qrels` for MTEB retrieval datasets
- Add the optional `mteb` extra
- *(dense)* Short aliases for MTEB datasets (e.g. `scidocs-qwen3-4b`), with `register_alias` and `resolve_name`

### 📚 Documentation

- Plans for MTEB dense datasets
- Document MTEB embeddings in the readme
- Plan and readme for the MTEB aliases

### ⚙️ Miscellaneous Tasks

- Add maintainer scripts to build and upload MTEB caches, and network tests

## [0.1.0] - 2026-10-10

### 🚀 Features

- *(dense)* Arbitrary pipelines on load
- *(graph)* Add graph module
- *(dense)* Add more dense datasets
- *(dense)* Store parse caches as zstd-compressed parquet
- *(dense)* Delete raw downloads once their cache exists
- *(graph)* Store local graphs as parquet files
- *(timeseries)* Add ATTIMO/MOMENTI time series cached as parquet
- *(timeseries)* Add Motiflets and MOMENTI plain text datasets
- *(timeseries)* Add a .tsf parser and the oikolab weather series
- *(timeseries)* Add FL010 through the optional wfdb extra

### 🐛 Bug Fixes

- *(dense)* Deduplication does not change row order
- *(test)* Fix failing test
- Download of files
- *(download)* Unique temporary files and wire-byte length check
- *(timeseries)* Tolerate ':' in the last .tsf attribute
- *(graph)* Use 64-bit adjacency offsets for graphs with 2^31+ edges

### 📚 Documentation

- Fix a typo in the readme
- Record the time series work and mark priority 3 as done

### 🚜 Refactor

- *(dense)* Change dataset registration
- *(dense)* Change the return type of the load function
- Share the parquet cache helpers between dense and graph

### ⚙️ Miscellaneous Tasks

- Lower minimum python version
- Update uv.lock
- Add agent's plan files
- Add github workflow
## [0.2.0] - 2026-10-11

### 🚀 Features

- Download dataset to user-specific cache
## [0.1.0] - 2026-10-10

### 🚀 Features

- *(dense)* Arbitrary pipelines on load
- *(graph)* Add graph module
- *(dense)* Add more dense datasets
- *(dense)* Store parse caches as zstd-compressed parquet
- *(dense)* Delete raw downloads once their cache exists
- *(graph)* Store local graphs as parquet files
- *(timeseries)* Add ATTIMO/MOMENTI time series cached as parquet
- *(timeseries)* Add Motiflets and MOMENTI plain text datasets
- *(timeseries)* Add a .tsf parser and the oikolab weather series
- *(timeseries)* Add FL010 through the optional wfdb extra

### 🐛 Bug Fixes

- *(dense)* Deduplication does not change row order
- *(test)* Fix failing test
- Download of files
- *(download)* Unique temporary files and wire-byte length check
- *(timeseries)* Tolerate ':' in the last .tsf attribute
- *(graph)* Use 64-bit adjacency offsets for graphs with 2^31+ edges

### 📚 Documentation

- Fix a typo in the readme
- Record the time series work and mark priority 3 as done

### 🚜 Refactor

- *(dense)* Change dataset registration
- *(dense)* Change the return type of the load function
- Share the parquet cache helpers between dense and graph

### ⚙️ Miscellaneous Tasks

- Lower minimum python version
- Update uv.lock
- Add agent's plan files
- Add github workflow
- Add changelog
