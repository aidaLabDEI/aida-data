## [0.4.0] - 2026-11-20

### 🚀 Features

- *(dense)* Add a couple of hosted datasets
- *(dense)* Cache every non-HDF5 dataset as parquet
- *(dense)* Derived cache for higgs-highlevel
- Convert parquet caches to HDF5
- Export library module
- *(cli)* Aida-data export, path, list and prune-hdf5
- *(cli)* Aida-data snippet prints a C++ (H5Easy) loader

### 📚 Documentation

- Document the command line and mark the export plan as implemented

### 🚜 Refactor

- *(dense)* Pure parsers and _ensure_cache
- *(graph, timeseries)* Split _cached into _ensure_cache and a read

### 🧪 Testing

- Network tests for aida-data export
## [0.3.0] - 2026-11-17

### 🚀 Features

- *(dense)* Optional row ids in Dataset and parquet cache
- *(mteb)* Encode MTEB tasks into the dense parquet cache
- *(dense)* Load mteb:<task>:<model> names and add load_qrels
- *(dense)* Short aliases for MTEB datasets

### 📚 Documentation

- Plans for MTEB dense datasets
- Document MTEB embeddings in the readme and changelog
- Mark the MTEB dense datasets plan as implemented
- Document MTEB aliases and mark the plan as implemented

### ⚙️ Miscellaneous Tasks

- Update changelog

### 💼 Other

- Mteb extra, maintainer scripts, network tests
- Upload_mteb requires a model tag and prints the hosted line
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
