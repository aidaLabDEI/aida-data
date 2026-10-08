# TODO: datasets used by AIDA Lab software not yet covered by `aida_data`

Survey of the repositories listed in `Software-list.md` (checked on
2026-10-08 by reading each repository's README, data folders and download
scripts). For each repository we list whether the data it uses is already
available in `aida_data`, and below we collect the missing datasets, grouped by
type.

Currently covered by `aida_data`:

- `aida_data.dense`: `landmark-nomic-768-normalized`,
  `imagenet-clip-512-normalized`, `agnews-mxbai-1024-euclidean`,
  `celeba-resnet-2048-cosine`, `simplewiki-openai-3072-normalized`,
  `deep-image-96-angular`, `mnist-784-euclidean`,
  `fashion-mnist-784-euclidean`, `glove-100-angular`, `gist-960-euclidean`,
  `nytimes-256-angular`, `sift-128-euclidean`, `pamap2`, `census` (ADBench),
  `ht`, `yandex-t2i`, `chem`, `densired-hard`;
- `aida_data.graph`: the 12 Sirius colored graphs (`abortion`, `brexit`,
  `citeseer`, `com-dblp`, `com-youtube`, `combined`, `obamacare`,
  `phy_citations`, `trivago-clicks`, `twitter_pol`, `uselections`,
  `walmart-trips`).

## 1. Coverage by repository

| Repository | Covered? | Notes |
|---|---|---|
| [Hephaestus](https://github.com/cecca/hephaestus) | yes | Uses ann-benchmarks/VIBE HDF5 files (example: `fashion-mnist-784-euclidean`). |
| [PANNA](https://github.com/cecca/panna) | yes | `pypanna/datasets.py` registers the same datasets as `aida_data.dense`. |
| [ATTIMO](https://github.com/Cecca/attimo) | no | Univariate time series (§3.1). |
| [MOMENTI](https://github.com/aidaLabDEI/MOMENTI-motifs) | no | Multivariate time series (§3.2). |
| [fair-clustering](https://github.com/Cecca/fair-clustering) | no | Colored tabular data (§2.1). Note: its `census1990` (USCensus1990) is not the ADBench `census` we have. |
| [streaming-fair-center-clustering](https://github.com/aidaLabDEI/streaming-fair-center-clustering) | no | §2.1 (HIGGS, PHONES, COVERTYPE, BEERS) and §8 (UBER). |
| [MACACO](https://github.com/Cecca/macaco) | no | §2.1 (Phones, Higgs) and §2.2 (Wikipedia, MusixMatch). |
| [DANNY](https://github.com/Cecca/danny) | partly | SIFT is the same SIFT1M base set as `sift-128-euclidean`, but DANNY preprocesses it differently. GloVe-Twitter-200 and the set datasets are missing (§2.2, §6). |
| [UGRAPH](https://github.com/Cecca/ugraph) | no | Uncertain graphs (§4.2). |
| [FewRS](https://github.com/leonardopellegrina/FewRS) | n/a | The repository only has a README so far; recheck when the code is published. |
| [Sirius](https://github.com/leonardopellegrina/Sirius) | yes | All 12 graphs with labels are registered. `guncontrol` only has community files and is not registered. |
| [PERCIS](https://github.com/Antonio-Cruciani/PERCIS) | partly | `abortion`, `combined` and `youtube` overlap with Sirius, but the percolation states and the remaining SNAP graphs are missing (§4.1). |
| [ALLSTAR](https://github.com/VandinLab/ALLSTAR) | no | TCGA BRCA data plus a PPI network (§9). |
| [MASTRO](https://github.com/VandinLab/MASTRO) | no | Tumor phylogenetic trees (§9). |
| [CPhyT-GNN](https://github.com/VandinLab/CPhyT-GNN) | no | Tumor phylogenetic trees with survival data (§9). |
| [UNCOVER](https://github.com/VandinLab/UNCOVER) | no | CCLE/Achilles example data (§9). |
| [DAMOKLE](https://github.com/VandinLab/DAMOKLE) | n/a | Code only, no data or data references in the repository. |
| [NoMAS](https://github.com/VandinLab/NoMAS) | no | TCGA survival and mutation data plus interaction networks (§9). |
| [SAKEIMA](https://github.com/VandinLab/SAKEIMA) | no | Sequencing reads (§10). |
| [SPRISS](https://github.com/VandinLab/SPRISS) | no | Sequencing reads (§10). |
| [TopKWY](https://github.com/VandinLab/TopKWY) | no | Labeled transactional data (§5.1) and labeled graph collections (§4.4). |
| [SPuManTE](https://github.com/VandinLab/SPuManTE) | no | Labeled transactional data (§5.1). |
| [PROMISE](https://github.com/VandinLab/PROMISE) | no | Sequential datasets (§5.2). |
| [VCRadSPM](https://github.com/VandinLab/VCRadSPM) | no | Sequential datasets (§5.2). |
| [MCRapper](https://github.com/VandinLab/MCRapper) | no | Transactional datasets (§5.1). |
| [gRosSo](https://github.com/VandinLab/gRosSo) | no | Sequence of sequential datasets built from Netflix Prize data (§5.2). |
| [CASPITA](https://github.com/VandinLab/CASPITA) | no | Path/trajectory data on an unknown network (§7). |
| [FSR](https://github.com/VandinLab/FSR) | no | Labeled tabular data (§2.3). |
| [SamRuLe](https://github.com/VandinLab/SamRuLe) | no | Binarized labeled tabular data (§2.3). |
| [RAveL](https://github.com/VandinLab/RAveL) | no | Boston housing plus synthetic data (§2.3). |
| [SubgraphMining_ICDE2023](https://github.com/VandinLab/SubgraphMining_ICDE2023) | n/a | Code only; it reads gSpan-format graph collections, but the repository names no datasets. It likely uses the same TU datasets as TopKWY (§4.4), to be checked against the paper. |
| [SILVAN](https://github.com/VandinLab/SILVAN) | no | SNAP/KONECT graphs (§4.1). |
| [PRESTO](https://github.com/VandinLab/PRESTO) | no | Temporal networks (§4.3). |
| [odeN](https://github.com/VandinLab/odeN) | no | Temporal networks (§4.3). |
| [ALDENTE](https://github.com/iliesarpe/ALDENTE) | n/a | Code only (toy graph in `snap-libs/examples/densemotifs`). The temporal networks used are only listed in the paper (arXiv 2406.10608), to be checked. |
| [Tonic](https://github.com/VandinLab/Tonic) | no | Graph streams and snapshot sequences (§4.1, §4.5). |
| [STEP](https://github.com/VandinLab/STEP) | no | Temporal networks (§4.3). |
| [PreSenS](https://github.com/VandinLab/PreSenS) | no | Timestamped point datasets split into snapshots (§8). |
| [silhouette-scalable](https://github.com/iliesarpe/ScalableSilhouetteComputation) | no | UCI/KDD dense datasets (§2.1) and Gowalla (§8). |
| [HotNet](https://github.com/raphael-group/hotnet) | no | Interaction networks and influence matrices (§9). |
| [HotNet2](https://github.com/raphael-group/hotnet2) | no | Pan-cancer heat scores and interaction networks (§9). |
| [CoMEt](https://github.com/raphael-group/comet) | no | TCGA mutation matrices (§9). |
| [ExaLT](https://github.com/fvandin/ExaLT) | n/a | Only toy example tables are included. |
| [TDA_hierarchical](https://github.com/aidaLabDEI/TDA_hierarchical) | no | ISTAT origin/destination matrices (§8). |

## 2. Dense / tabular datasets

### 2.1 Dense vectors for clustering (often with categorical "color" attributes)

| Dataset | Source | Used by | Notes |
|---|---|---|---|
| HIGGS | <https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz> | streaming-fair, MACACO, FSR, SamRuLe (`higgs_tab`) | 11M x 28. Reused widely, so it is a high priority. |
| PHONES (Heterogeneity Activity Recognition) | <https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip> | streaming-fair, MACACO | Accelerometer data. |
| Covertype | <https://archive.ics.uci.edu/dataset/31/covertype> | streaming-fair, FSR (`covtype.data`) | Transactional version in §5.1. |
| BEERS (BeerAdvocate reviews) | <https://www.kaggle.com/datasets/thedevastator/1-5-million-beer-reviews-from-beer-advocate> | streaming-fair | Kaggle (needs login); preprocessed with `beers.py` into 27 categories. |
| Randomized/BLOBS variants | [Google Drive](https://drive.google.com/drive/folders/1YraBr_UZhe9sNAXeCGSiTX9hPQdQDIWW) | streaming-fair | Shuffled or synthetic; could be generated instead. |
| adult | <https://archive.ics.uci.edu/static/public/2/adult.zip> | fair-clustering, FSR, SamRuLe | |
| athlete (Olympic history) | <https://github.com/rgriff23/Olympic_history/raw/master/data/athlete_events.csv> | fair-clustering | |
| diabetes (130 US hospitals) | <https://archive.ics.uci.edu/static/public/296/diabetes+130+us+hospitals+for+years+1999+2008.zip> | fair-clustering | |
| census1990 / census1990_age | <https://web.archive.org/web/20170711094723/https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt> | fair-clustering | Not the ADBench `census`. |
| creditcard (default of credit card clients) | <https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip> | fair-clustering, silhouette | |
| 4area, reuter_50_50 (c50), victorian, bank | `https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/{4area,c50,victorian,bank_categorized}.csv` | fair-clustering | |
| hmda (2022 modified LAR) | <https://s3.amazonaws.com/cfpb-hmda-public/prod/dynamic-data/combined-mlar/2022/header/2022_combined_mlar_header.zip> | fair-clustering | Large. |
| breast (Wisconsin diagnostic) | <https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip> | silhouette | |
| wine quality | <https://archive.ics.uci.edu/static/public/186/wine+quality.zip> | silhouette | |
| shuttle (Statlog) | <https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip> | silhouette | `shuttle.trn.Z` must be uncompressed. |
| RT-IoT2022 | <https://archive.ics.uci.edu/static/public/942/rt-iot2022.zip> | silhouette | |
| BioKDD (KDD Cup 2004 protein homology) | <https://kdd.org/cupfiles/KDDCupData/2004/data_kddcup04.tar.gz> | silhouette | `bio_train.dat` |
| RNA-seq (BMC Bioinf. 7:173 supplementary) | <https://static-content.springer.com/esm/art%3A10.1186%2F1471-2105-7-173/MediaObjects/12859_2005_912_MOESM6_ESM.gz> | silhouette | Needs manual cleaning (see `research/data/links.txt`). |
| MetroPT-3 | <https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip> | silhouette | |
| Household power consumption | <https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip> | silhouette | |

### 2.2 Embeddings / text-derived vectors

| Dataset | Source | Used by | Notes |
|---|---|---|---|
| GloVe Twitter 27B, 200d | <https://nlp.stanford.edu/data/glove.twitter.27B.zip> | DANNY (`Glove-27-200`) | Same data as ann-benchmarks `glove-200-angular`, which could be registered instead. |
| GloVe 6B | <http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip> | MACACO | Used to embed Wikipedia articles. |
| Wikipedia (LDA topics + embeddings) | `https://dumps.wikimedia.org/enwiki/20210120/enwiki-20210120-pages-articles-multistream.xml.bz2` | MACACO | Heavy preprocessing; preprocessed copies were at `https://www.inf.unibz.it/~ceccarello/data` (check if still online). |
| MusixMatch (MSD lyrics + tagtraum genres) | `http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_{train,test}.txt.zip`, <http://www.tagtraum.com/genres/msd_tagtraum_cd2.cls.zip> | MACACO | Bag of words with genre colors. |
| SIFT1M (texmex), DANNY variant | `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` | DANNY (`SIFT-100nn-0.5`) | Base vectors already available via `sift-128-euclidean`; only DANNY's preprocessing is missing. |

### 2.3 Labeled tabular data (subgroup discovery, rule lists, causal discovery)

| Dataset | Source | Used by | Notes |
|---|---|---|---|
| FSR collection: adult, mushroom, bank-additional-full, theorem-prover, abalone, TCGA-brain-cancer, covtype, gisette, HIGGS, SUSY, kdd-cup, cancer-rna-seq-merge, mnist | <https://tinyurl.com/FSRdatasets> (zip) | FSR | Most are UCI; the zip is the easiest source. |
| SamRuLe collection: mushroom, susy, phishing, adult, bank, higgs, ijcnn1, a9a (`*_tab.csv`) | <http://tinyurl.com/SamRuLedata> (zip) | SamRuLe | Binarized tabular format. |
| Boston housing | in repo: `datasets/real_data/housing.csv` | RAveL | Ethically problematic dataset; consider whether to include it. |

## 3. Time series

### 3.1 Univariate

| Dataset | Source | Used by | Notes |
|---|---|---|---|
| ASTRO, ECG, freezer, GAP | figshare files 36982360, 36982384, 36982390, 36982396 (`https://figshare.com/ndownloader/files/<id>`); also in `data/` of the repo | ATTIMO (`pyattimo.load_dataset`) | One value per line, gzipped. |
| HumanY | `data/HumanY.txt.gz` in the ATTIMO repo (31 MB) | ATTIMO | Also on the figshare collection <https://figshare.com/articles/dataset/Datasets/20747617>. |
| ecg-heartbeat-av, insect_b, case1, insect15 | in the ATTIMO repo (`data/`, `docs/`) | ATTIMO | Small examples, low priority. |
| Whales, VCAB (noised), synthetic `synth-w100-*` | not published (commented out in `experiments.py`) | ATTIMO | Synthetic ones could be generated. |
| steamgen | <https://zenodo.org/record/4273921/files/STUMPY_Basics_steamgen.csv?download=1> | ATTIMO, MOMENTI | Multivariate (4 dims). |
| Motiflets datasets: Arrhythmia, penguin, ASTRO, GAP, dishwasher, EEG-sleep (npo141), PAMAP | `https://github.com/patrickzib/motiflets/raw/refs/heads/pyattimo/datasets/...` | ATTIMO (`pyattimo/bench.py`) | PAMAP here is a time-series view; `aida_data.dense.pamap2` treats it as points. |
| MOMP collection (`*.mat`) | `https://github.com/patrickzib/motiflets/raw/refs/heads/pyattimo_refactor/datasets/momp/{name}.mat` | ATTIMO (`pyattimo/check.py`) | |

### 3.2 Multivariate

| Dataset | Source | Used by | Notes |
|---|---|---|---|
| FOETAL_ECG, evaporator | in the MOMENTI repo `Datasets/` (from the DaISy database) | MOMENTI | |
| oikolab weather | in the MOMENTI repo (`.tsf`, Monash forecasting archive) | MOMENTI | Parser in `external_dependecies/data_loader.py`. |
| RUTH | in the MOMENTI repo (`Datasets/RUTH.csv`, 12 MB) | MOMENTI | Origin to be checked. |
| CLEAN_House1 (REFIT electrical load) | not in the repo | MOMENTI | |
| whales | not in the repo; built from NOAA SanctSound clips (`Datasets/Whale.ipynb`) | MOMENTI | |
| quake | not in the repo; IU.ANMO seismic trace via ObsPy (`Seism_retr.py`) | MOMENTI | |
| FL010 | PhysioNet record (`.hea` in the repo, read with `wfdb`) | MOMENTI | |

## 4. Graphs

### 4.1 Static graphs (SNAP / KONECT / Network Repository)

| Dataset | Source | Used by |
|---|---|---|
| p2p-Gnutella31, cit-HepTh, cit-HepPh, soc-Epinions1, wiki-Vote, wiki-topcats, email-EuAll, wiki-Talk, soc-LiveJournal1, soc-pokec (directed) | `http://snap.stanford.edu/data/<name>.txt.gz` | SILVAN (cit-HepPh, Epinions, Gnutella31 also PERCIS; LiveJournal also Tonic) |
| com-amazon, email-Enron, ca-GrQc, com-youtube, com-dblp, ca-AstroPh (undirected) | `http://snap.stanford.edu/data/...` | SILVAN (Enron, AstroPh also PERCIS) |
| wikipedia_link_en, actor-collaboration | `http://konect.cc/files/download.tsv.<name>.tar.bz2` | SILVAN, Tonic (actors) |
| musae-facebook, web-NotreDame, web-Google, web-BerkStan, flickr, soc-Slashdot | SNAP | PERCIS |
| PERCIS bundle (graphs + percolation states, including `_lcc_in_50` variants, abortion, combined, guncontrol, youtube) | [mega.nz link](https://mega.nz/file/ws4xSJLJ#eCTj57leI_3HgxCTlrM1kl056sWLLYhmfN5okj1erSU) (`download_datasets.txt`) | PERCIS |
| cit-Patents | <https://snap.stanford.edu/data/cit-Patents.html> | Tonic |
| edit-enwikibooks, soc-youtube-growth | <https://networkrepository.com/> | Tonic (wikibooks preprocessed copy in repo) |
| sx-stackoverflow | <http://konect.cc/networks/sx-stackoverflow> | Tonic |
| Twitter-merged (WWW2010) | <https://anlab-kaist.github.io/traces/WWW2010> | Tonic (1.5B edges) |

Note: SNAP `com-dblp` and `com-youtube` have the same names as the Sirius
datasets, but they are different files (uncolored SNAP versions). `com-*`
loaders could share code.

### 4.2 Uncertain (edge-probability) graphs

| Dataset | Source | Used by |
|---|---|---|
| Collins 2007, Gavin 2006, Krogan 2006 core/extended (+ LCC versions) | in the UGRAPH repo `Reproducibility/Data/proteins/` (from <http://www.paccanarolab.org/static_content/clusterone/cl1_datasets.zip>) | UGRAPH |
| TAP core + MIPS ground truth + MCL clusters | in the UGRAPH repo (from <http://tap.med.utoronto.ca/exttap/>) | UGRAPH |
| DBLP co-authorship (probabilities from # collaborations) | in the UGRAPH repo `Reproducibility/Data/dblp/dblp-vldb-publication.txt.bz2` | UGRAPH |

### 4.3 Temporal networks (timestamped edges)

| Dataset | Source | Used by |
|---|---|---|
| email-Eu-core-temporal, sx-mathoverflow, sx-superuser | SNAP (copies in `examples/test-graphs/`) | PRESTO |
| EquinixChicago (A/B) | [Google Drive](https://drive.google.com/drive/folders/1HXMEO4wwMOT1H9icwiP6siQTp2sm5pgA), built from CAIDA pcap traces | PRESTO, STEP (STEP adds edges with `synthEquinix`, >70 GB) |
| CollegeMsg, email, sms (remapped) | in the odeN repo `examples/datasets/` | odeN |
| sx-stackoverflow (temporal) | <https://snap.stanford.edu/data/sx-stackoverflow.html> | STEP |
| temporal-bitcoin | <https://www.cs.cornell.edu/~arb/data/temporal-bitcoin/> | STEP |
| temporal-reddit-reply | <https://www.cs.cornell.edu/~arb/data/temporal-reddit-reply/> | STEP |
| ALDENTE datasets | listed in the paper only (arXiv 2406.10608) | ALDENTE |

### 4.4 Collections of labeled graphs

| Dataset | Source | Used by |
|---|---|---|
| AIDS, BZR, COX2, DD, DHFR, ENZYMES, MUTAG, Mutagenicity, NCI1, NCI109, Tox21_AHR | TU Dortmund graph benchmarks; converted copies in `topkwy-subgraph/datasets/datasets.zip` | TopKWY (probably also SubgraphMining_ICDE2023) |

### 4.5 Snapshot sequences / graph streams

| Dataset | Source | Used by |
|---|---|---|
| Oregon-1 (9 snapshots) | <https://snap.stanford.edu/data/Oregon-1.html> | Tonic |
| as-caida (122 snapshots) | <https://snap.stanford.edu/data/as-caida.html> | Tonic (one preprocessed snapshot in repo) |
| as-733 (733 snapshots) | <https://snap.stanford.edu/data/as-733.html> | Tonic |
| Twitter snapshots (4) | <https://anlab-kaist.github.io/traces/WWW2010> | Tonic |

## 5. Pattern mining datasets

### 5.1 Transactional (itemsets), optionally with class labels

| Dataset | Source | Used by |
|---|---|---|
| a9a, covtype, ijcnn1, phishing, svmguide3, susy, cod-rna, breast-cancer, mushroom (LIBSVM-derived, with `.labels`) | `datasets.zip` in TopKWY (`TopKWY/datasets/`), SPuManTE and MCRapper | TopKWY, SPuManTE, MCRapper |
| accidents, bms-pos, bms-web1, bms-web2, chess, connect, pumsb-star, retail, T10I4D100K, T40I10D100K (FIMI) | same zips (TopKWY has labeled `_new` versions) | TopKWY, MCRapper, SPuManTE (retail) |

These are the same files across the three repositories (MCRapper's
`susy.dat` alone is 315 MB). A single `aida_data.transactions` module could
serve them all.

### 5.2 Sequential (SPMF format)

| Dataset | Source | Used by |
|---|---|---|
| BIBLE, BIKE, FIFA, LEVIATHAN, SIGN | in repo `data/` | PROMISE, VCRadSPM |
| BMS1, BMS2, KOSARAK, MSNBC | in repo `data/` | VCRadSPM |
| Netflix sequence of datasets | <https://www.kaggle.com/netflix-inc/netflix-prize-data> (`combined_data_1..4`), converted with `NetflixDataset.java` | gRosSo |

## 6. Set data (similarity joins)

| Dataset | Source | Used by |
|---|---|---|
| Orkut group memberships | <http://socialnetworks.mpi-sws.mpg.de/data/orkut-groupmemberships.txt.gz> | DANNY |
| LiveJournal group memberships | <http://socialnetworks.mpi-sws.mpg.de/data/livejournal-groupmemberships.txt.gz> | DANNY |
| ssjoin benchmark scripts (AOL, ...) | <http://ssjoin.dbresearch.uni-salzburg.at/datasets.html> | DANNY (`datasets/scripts/`) |

## 7. Paths / trajectories on networks

| Dataset | Source | Used by |
|---|---|---|
| BIKE (LA Metro bike share 2019), BIKE10, BIKE20, NEWBIKE (2020) | <https://bikeshare.metro.net/about/data/>; generated copies in `reproducibility/CASPITA/data/` | CASPITA |
| WIKI (Wikispeedia finished paths) | <https://snap.stanford.edu/data/wikispeedia.html>; generated copy in repo | CASPITA |
| FLIGHT (DB1B coupon 2019) | <https://www.transtats.bts.gov/> (DB1BCoupon 2019 Q1–Q4) | CASPITA |

## 8. Spatio-temporal points and origin/destination data

| Dataset | Source | Used by | Notes |
|---|---|---|---|
| Twitter geospatial | <https://archive.ics.uci.edu/dataset/1050/twitter+geospatial+data> | PreSenS | 14M points, daily snapshots. |
| Intel Lab sensor data | <https://db.csail.mit.edu/labdata/labdata.html> | PreSenS | |
| Taxi (Porto, ECML/PKDD 2015) | <https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015> | PreSenS | Trajectories. |
| NYC TLC yellow taxi (monthly / yearly) | <https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data>, <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> | PreSenS | Up to 743M rows; `data/download_and_parse_parquet.py`. |
| UBER (TLC FOIL) | <https://github.com/fivethirtyeight/uber-tlc-foil-response> | streaming-fair | Merged with `merger_uber.py`. |
| Gowalla check-ins | <https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz> | silhouette | Lat/lon columns only. |
| ISTAT commuting O/D matrices (2011 census) | <https://www.istat.it/storage/cartografia/matrici_pendolarismo/matrici-pendolarismo-sezione-censimento-2011.zip> | TDA_hierarchical | Hierarchical O/D; synthetic trees generated by scripts. |

## 9. Cancer genomics

| Dataset | Source | Used by |
|---|---|---|
| TCGA BRCA: clinical, LOH, MAF, methylation, histology, aggregated matrix | in the ALLSTAR repo `data/` | ALLSTAR |
| Reactome FI network (`FIsInGene_122921_with_annotations.txt`) | in the ALLSTAR repo `PPI/` | ALLSTAR |
| AML and NSCLC/lung tumor phylogenies | in the MASTRO repo `data/` | MASTRO |
| Breast cancer and AML phylogenies; breast (Razavi) and TRACERx lung phylogenies with clinical data | in the CPhyT-GNN repo `data/` | CPhyT-GNN |
| CCLE mutations/CNAs, Achilles RNAi, target profiles (`.gct`) | in the UNCOVER repo `Example/` | UNCOVER |
| TCGA GBM, LUSC, OV survival + mutations | in the NoMAS repo `datasets/` | NoMAS |
| HINT+HI2012, iRefIndex networks | in the NoMAS repo `networks/`, HotNet2 `paper/data/networks/` | NoMAS, HotNet2 |
| MultiNet, iRefIndex9 networks; pan-cancer (pan12) heat scores | in the HotNet2 repo `paper/data/` | HotNet2 |
| HPRD, iRefIndex edge lists + influence matrices | edge lists in the HotNet repo; matrices from <http://compbio.cs.brown.edu/projects/hotnet/> | HotNet |
| AML, BRCA, GBM, STAD mutation matrices (`.m2`) | in the CoMEt repo `example_datasets/` | CoMEt |

These are small and domain-specific. They could go into a module of their own
(for example `aida_data.genomics`) or stay out of scope.

## 10. Sequencing reads (k-mer counting)

| Dataset | Source | Used by |
|---|---|---|
| HMP stool and tongue-dorsum samples (e.g. SRS024075, SRS024388, SRS011239, SRS075404, SRS043663, SRS062761) | `ftp://public-ftp.ihmpdcc.org/Illumina/{stool,tongue_dorsum}/<id>.tar.bz2` | SAKEIMA, SPRISS |
| GOS samples GS002–GS051 (`*.fa`) | <https://www.imicrobe.us/#/samples> | SPRISS |
| Maize B73 / Mo17 (SRP082260) | <https://www.ncbi.nlm.nih.gov/sra/?term=SRP082260> | SPRISS |
| GIAB NA12878 (75x) + hg19 + VCFs + confident regions | <https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/>, <http://cb.csail.mit.edu/cb/lava/data/hg19.fa.gz>, Google Drive links in `data_preparation.txt` | SPRISS (genotyping application) |

These are very large FASTQ/FASTA files, so they are probably out of scope
except as download helpers.

## 11. Suggested priorities

1. ~~**Dense** (`aida_data.dense`): HIGGS, PHONES, covertype, adult and the
   fair-clustering datasets (they need a "colors" field, as in
   `aida_data.graph`), plus the silhouette UCI datasets.~~ Done on
   2026-10-08 (see `2026-10-08-colored-dense-datasets.md`), except hmda,
   BEERS, BLOBS and RNA-seq.
2. **Graphs** (`aida_data.graph`): add uncolored SNAP/KONECT loaders for the
   SILVAN/PERCIS/Tonic graphs; consider a separate temporal edge list type
   (`src dst timestamp`) for PRESTO/odeN/STEP.
3. **Time series**: new `aida_data.timeseries` module for the ATTIMO and
   MOMENTI datasets.
4. **Transactions / sequences**: a module for the FIMI/LIBSVM transactional
   datasets and the SPMF sequential datasets (all already hosted in the lab
   repositories).
5. Lower priority: spatio-temporal points, trajectories, set data, genomics and
   sequencing reads.
