# MulDR

Source code for "Structured Multi-view Relational Learning for Drug Response Prediction".

MulDR is a drug response prediction model that integrates multi-omics cell-line features and molecular drug representations for cell-drug response regression.

## Requirements

The main dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The code was tested with Python 3.10, PyTorch 2.1.2, pandas 2.3.3, NumPy 1.26.3, SciPy 1.15.3, and RDKit.

## Files

### Data

Place the required data files under `data/`:

```text
data/
  CELL/
    geo_expression_cosmic.csv
    geo_methylation_cosmic.csv
    geo_mutation_cosmic.csv
    pathway_cosmic.csv
  drug_info/
    drug_info.csv
  GDSC2_IC50.csv
```

## Usage

Before training, generate the functional group pattern file:

```bash
python scripts/gen_new_fg_groups.py
```

Then build the motif vocabulary and molecular graph cache:

```bash
python build_motif_vocab_and_cache.py
```

Train and evaluate the model:

```bash
python train.py
```

Training logs, checkpoints, preprocessing files, and cache files are generated locally and are not part of the source release.
