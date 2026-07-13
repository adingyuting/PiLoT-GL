# PiLoT-GL

This repository contains the clean training code for **PiLoT-GL**, a dynamic graph representation model with:

- short/mid/long temporal graph views
- a shared temporal graph encoder
- scale-specific LoRA edge-logit residuals
- time-prior-conditioned LoRA gates
- scale-attention fusion
- power-law weighted GLSL with multi-scale disagreement modulation

## Repository Layout

```text
.
|-- constant.py
|-- loss.py
|-- run.py
|-- train.py
|-- configs/
|   |-- bitcoin_alpha.yaml
|   |-- bitcoin_otc.yaml
|   |-- digg.yaml
|   |-- ia_reality_call.yaml
|   |-- internet.yaml
|   |-- ppin.yaml
|   |-- wiki_eo.yaml
|   \-- wiki_gl.yaml
|-- data/
|   \-- .gitkeep
|-- data_process/
|   |-- constant.py
|   |-- process.py
|   \-- process_bitcoin.py
|-- models/
|   |-- cdpss.py
|   |-- layers.py
|   \-- model.py
\-- utils/
    |-- DataLoader.py
    |-- EarlyStopping.py
    |-- load_configs.py
    \-- util.py
```

Runtime outputs are intentionally ignored by git: dataset files under `data/`, `logs/`, `results/`, and `saved_models/`.

## Installation

```bash
pip install -r requirements.txt
```

## Data

Place each dataset under `data/<dataset_name>/<dataset_name>.mat`, for example:

```text
data/
  wiki_gl/
    wiki_gl.mat
```

The repository includes an empty `data/` directory placeholder. Put downloaded or preprocessed datasets there before training.

Dataset sources:

| Dataset | Source |
|---|---|
| `ia_reality_call` | [Reality Mining](http://realitycommons.media.mit.edu/realitymining.html) |
| `internet` | [UCLA Internet Topology Collection](http://irl.cs.ucla.edu/topology/) |
| `ppin` | [DPPIN-Babu](https://github.com/DongqiFu/DPPIN/tree/main/DPPIN-Babu) |
| `wiki_eo` | [KONECT wiki_talk_eo](http://konect.cc/networks/wiki_talk_eo) |
| `wiki_gl` | [KONECT wiki_talk_gl](http://konect.cc/networks/wiki_talk_gl) |
| `digg` | [KONECT munmun_digg_reply](http://konect.cc/networks/munmun_digg_reply) |
| `bitcoin_alpha` | [SNAP Bitcoin Alpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) |
| `bitcoin_otc` | [SNAP Bitcoin OTC](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) |

You can also point the code to an external data directory:

```bash
export PILOT_GL_DATA_DIR=/path/to/data
```

On Windows PowerShell:

```powershell
$env:PILOT_GL_DATA_DIR="D:\path\to\data"
```

## Run

Use the default configuration:

```bash
python run.py
```

Select a dataset configuration from `configs/`:

```bash
python run.py --dataset_name wiki_eo
```

Or pass a config file explicitly:

```bash
python run.py --config configs/wiki_eo.yaml
```

Override training parameters:

```bash
python run.py --set epochs=20 --set cuda=False
```

Logs and summaries are written to `logs/<dataset_name>/`; model checkpoints are written to `saved_models/<dataset_name>/`.
