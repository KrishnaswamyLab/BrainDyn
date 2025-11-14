# BrainDyn

## Quick start

### Prerequisites
- Python 3.8+
- git

### Install dependencies (recommended: uv)
Note: "uv" is a developer tool for creating a venv and syncing dependencies — it is NOT a data file.

Recommended (uv):
```bash
uv venv
source .venv/bin/activate
uv sync
```

Alternative (pip):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate the dataset
Place all runtime / dataset files in a `data/` folder at the repo root. For running the synthetic data using the kuramoto model you can run:

```bash
python kuramoto_brain_multirhythm.py
```

### Run the main pipeline
```bash
python main.py
```
