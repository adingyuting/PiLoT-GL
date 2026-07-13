# Data Preprocessing

This folder contains preprocessing scripts for converting raw dynamic graph data into the `.mat` files consumed by the training pipeline.

Expected processed data layout:

```text
data/
  <dataset_name>/
    <dataset_name>.mat
```

Bitcoin datasets can be processed from their raw CSV files with:

```bash
python data_process/process_bitcoin.py
```
