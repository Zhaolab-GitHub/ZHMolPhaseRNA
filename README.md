# ZHMolPhaseRNA

## 1. Overview

**ZHMolPhaseRNA is a sequence-based machine-learning framework for predicting RNAs involved in liquid–liquid phase separation (LLPS).** It integrates pretrained ERNIE-RNA representations with handcrafted sequence features and LightGBM models.

**We also provide an online prediction server for convenient use without local installation:**

**[Click to access the ZHMolPhaseRNA online server](http://zhaoserver.com.cn/ZHMolPhaseRNA/ZHMolPhaseRNA.html)**

This repository provides prediction and training code, pretrained models, datasets, and source data for reproducing the main figures.

## 2. Installation

ZHMolPhaseRNA supports CPU-based prediction and has been configured for Linux with Python 3.10.

```bash
git clone https://github.com/Zhaolab-GitHub/ZHMolPhaseRNA.git
cd ZHMolPhaseRNA

conda env create -f environment.yml
conda activate llps-rna
```

## 3. Download Pretrained Model

The 50 trained LightGBM models are already included in the repository. Only the pretrained ERNIE-RNA encoder weights need to be downloaded.

```bash
curl --fail --location --retry 3 \
  'https://huggingface.co/Taykhoom/ERNIE-RNA/resolve/aeea0c5af5f0734c4e15d974f6d95704fa2d7494/pytorch_model.bin' \
  --output models/ernie_rna/pytorch_model.bin
```

The ERNIE-RNA encoder is based on the [original ERNIE-RNA implementation](https://github.com/Bruce-ywj/ERNIE-RNA).

## 4. Usage

### Prepare Input Sequences

Provide RNA sequences in FASTA format. An example input file is available at `data/example.fasta`.

```fasta
>RNA_1
AUGCAUGCAUGCAUGC
>RNA_2
GGGAAAUUUCCCGGG
```

Sequences should contain A, C, G, and U nucleotides. Lowercase letters and T are automatically normalized; ambiguous nucleotides are not supported.

### Predict RNA LLPS

Run the prediction script:

```bash
python predict.py \
  --fasta data/example.fasta \
  --threads 4 \
  --format tsv \
  --output output/predictions.tsv
```

The results include a `mean_probability` score averaged across 50 LightGBM models. The default classification threshold is 0.5.

To obtain individual model scores, add `--include-model-scores`.

An example output is provided in `results/example_predictions.tsv`.

## 5. Reproducing Model Training

The original training dataset, feature matrix, and ten-dataset five-fold cross-validation splits are included in `data/`.

To reproduce model training:

```bash
python train.py \
  --threads 16 \
  --output output/training
```

This retrains the 50 LightGBM models and generates cross-validation predictions and evaluation metrics.

The original feature matrix is provided in `data/training/X_899.npy`. Reproducing training from this matrix does not require downloading the ERNIE-RNA weights.

## 6. Datasets and Figure Reproduction

The repository includes:

- **Training data:** 364 Reviewed LLPS-associated RNAs and 3,640 unlabeled RNAs, organized into ten length-matched datasets.
- **Cross-validation:** Original group-aware five-fold data splits, predictions, and evaluation results.
- **Large-scale applications:** RNA ID lists for 55,919 HTP-supported RNAs and 179,602 RefSeq background RNAs.
- **Figure data:** Source data and available plotting scripts for Figures 2–6.

Unlabeled RNAs are used as background samples and should not be interpreted as experimentally confirmed non-LLPS RNAs.

The plotting scripts are provided in the corresponding `Figure2/`–`Figure6/` directories. For example:

```bash
python Figure2/plot_figure2.py
python Figure3/plot_figure3.py
python plot_cv.py --output output/cv_figures
```

Additional figure-specific scripts and source files are available in their respective directories.

## Contact

For questions or suggestions regarding ZHMolPhaseRNA, please contact:

Chengwei Zeng: [cwzengwuhan@mails.ccnu.edu.cn](mailto:cwzengwuhan@mails.ccnu.edu.cn)
