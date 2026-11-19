# LLPS-RNA

ZHMolPhase_RNA is a sequence-based machine-learning framework for identifying LLPS-associated RNAs.

## 1. Installation

Linux, CPU; prediction and training use the same environment.

```bash
conda env create -f environment.yml
conda activate llps-rna
```

Keep `requirements.txt` beside `environment.yml`. Use the specified Python 3.10 and zlib 1.3.1 versions to preserve feature calculations.

## 2. Files

- `data/`: training FASTA, original features, cross-validation splits and large-scale ID lists.
- `models/`: 50 LightGBM models, fixed parameters and ERNIE-RNA configuration/code.
- `results/`: original OOF predictions, metrics, and curves.

## 3. Download ERNIE-RNA

LightGBM models are included. Download only the matching ERNIE-RNA weights:

```bash
curl --fail --location --retry 3 \
  'https://huggingface.co/Taykhoom/ERNIE-RNA/resolve/aeea0c5af5f0734c4e15d974f6d95704fa2d7494/pytorch_model.bin' \
  --output models/ernie_rna/pytorch_model.bin
```

Keep the supplied configuration and tokenizer files. The encoder is the [Taykhoom conversion](https://huggingface.co/Taykhoom/ERNIE-RNA) of [ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA); the upstream license notice is retained in `models/ernie_rna/`.

## 4. Prediction

```bash
python predict.py --fasta data/example.fasta --threads 4 \
  --format tsv --output output/predictions.tsv
```

FASTA and gzip FASTA are supported. Use unique IDs and A/C/G/U sequences; lowercase and T are normalized, while N and other ambiguous bases are rejected. Add `--include-model-scores` to save all 50 scores.

`mean_probability` is the mean of the 50 model scores, with a default classification threshold of 0.5. It is not a calibrated probability of experimental LLPS. Example output: `results/example_predictions.tsv`.

## 5. Reproduce Training

Use the original `4004 × 899` feature matrix and fixed splits included in `data/`:

```bash
python train.py --threads 16 --output output/training
```

This retrains all 50 LightGBM models using the original cross-validation splits. Models, OOF predictions and metrics are saved to the output directory. Use a new directory for each run. ERNIE-RNA weights are not needed for this step.

At completion, `MATCH` or `DIFFERENT` reports agreement with the original OOF scores and metrics (absolute tolerance 1e-12). Original/reproduced metric values are printed and saved in `summary.tsv`; details are included in `training_report.json`. A partial run compares only its selected folds.

To recompute features from FASTA instead:

```bash
python extract_features.py --fasta data/training/combined.fasta \
  --output output/features --threads 4
python train.py --features output/features --threads 16 \
  --output output/training_recomputed
```

CPU feature extraction can be slow and can differ slightly from the original GPU embeddings. Use the supplied feature matrix to reproduce the original training scores.

## 6. Datasets

| Collection | RNA count | Included format |
| --- | ---: | --- |
| Training/CV | 364 Reviewed positive + 3,640 unlabeled = 4,004 | FASTA, features and original ten-dataset/five-fold splits |
| HTP-supported application | 55,919 | `data/large_scale/HTP_supported_RPS_ids.txt` |
| RefSeq background application | 179,602 | `data/large_scale/RefSeq_background_ids.txt` |

Unlabeled RNAs are not experimentally validated non-LLPS controls. Background `U_...` IDs are project-internal identifiers, not RefSeq accessions. Full large-scale FASTA files are distributed separately.

## 7. Reproduce Curves

```bash
python plot_cv.py --output output/cv_figures
```
