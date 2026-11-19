#!/usr/bin/env python3
"""Reproduce the frozen 10 x 5 LightGBM benchmark without tuning or refitting."""
import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
METRICS = ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']


def read_table(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, float_precision='round_trip')


def load_inputs(features=None):
    """Align features by RNA ID and retain the original column order and folds."""
    metadata = read_table(ROOT / 'data/training/metadata.tsv')
    if (len(metadata) != 4004 or not metadata.RNA_ID.is_unique
            or not np.array_equal(metadata.row_idx, np.arange(4004))
            or metadata.label.value_counts().to_dict() != {0: 3640, 1: 364}):
        raise ValueError('Expected the fixed union of 364 P and 3640 U.')
    folder = Path(features).resolve() if features else ROOT / 'data/training'
    columns = (folder / 'feature_columns_899.txt').read_text().splitlines()
    expected = (ROOT / 'models/feature_columns_899.txt').read_text().splitlines()
    if len(columns) != 899 or columns != expected:
        raise ValueError('Feature column order must be ERNIE768 then Handcrafted131.')
    x = np.load(folder / 'X_899.npy', allow_pickle=False)
    if x.shape != (4004, 899) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Expected a finite 4004 x 899 float32 matrix.')
    feature_md = read_table(folder / 'metadata.tsv')
    if (len(feature_md) != 4004 or not feature_md.RNA_ID.is_unique
            or not np.array_equal(feature_md.row_idx, np.arange(4004))):
        raise ValueError('Invalid feature row map.')
    ordered = feature_md.set_index('RNA_ID').loc[metadata.RNA_ID]
    for col in ['length']:
        if not np.array_equal(ordered[col], metadata[col]):
            raise ValueError('Feature identity mismatch: ' + col)
    x = x[ordered.row_idx.to_numpy(dtype=int)]
    frames, background_ids, positive_ids = [], set(), None
    for d in range(1, 11):
        folder_cv = ROOT / f'data/cross_validation/dataset_{d:02d}'
        md = read_table(folder_cv / 'members.tsv')
        if (len(md) != 728 or not md.RNA_ID.is_unique
                or md.label.value_counts().to_dict() != {1: 364, 0: 364}
                or set(md.fold_id) != set(range(1, 6)) or not md.dataset_id.eq(d).all()
                or not md.feature_row.between(0, 4003).all()):
            raise ValueError(f'Invalid dataset {d}.')
        ordered_md = metadata.iloc[md.feature_row]
        for col in ['RNA_ID', 'label', 'fold_id']:
            if not np.array_equal(md[col], ordered_md[col]):
                raise ValueError(f'Dataset {d} row mismatch: {col}')
        if not np.array_equal(md.group_id, ordered_md.global_group_id):
            raise ValueError('Dataset must use the final global similarity groups.')
        positives = set(md.loc[md.label.eq(1), 'RNA_ID'])
        negatives = set(md.loc[md.label.eq(0), 'RNA_ID'])
        if positive_ids is not None and positives != positive_ids:
            raise ValueError('All datasets must share the same 364 positives.')
        if negatives & background_ids:
            raise ValueError('Unlabeled backgrounds must be mutually exclusive.')
        positive_ids = positives
        background_ids |= negatives
        for fold in range(1, 6):
            tr, va = md.loc[md.fold_id.ne(fold)], md.loc[md.fold_id.eq(fold)]
            if set(tr.group_id) & set(va.group_id):
                raise ValueError('Similarity-group leakage across folds.')
            for part, rows in [('train', tr), ('val', va)]:
                ids = (folder_cv / f'fold_{fold}_{part}_ids.txt').read_text().splitlines()
                if ids != rows.RNA_ID.tolist():
                    raise ValueError('Published split IDs/order differ from membership table.')
        frames.append(md)
    if metadata.groupby('global_group_id').fold_id.nunique().max() != 1:
        raise ValueError('A global similarity group spans multiple folds.')
    return metadata, x, frames


def metrics(labels, scores):
    predicted = np.asarray(scores) >= 0.5
    return dict(AUROC=float(roc_auc_score(labels, scores)),
                AUPRC=float(average_precision_score(labels, scores)),
                Accuracy=float(accuracy_score(labels, predicted)),
                Precision=float(precision_score(labels, predicted, zero_division=0)),
                Recall=float(recall_score(labels, predicted, zero_division=0)),
                F1=float(f1_score(labels, predicted, zero_division=0)),
                MCC=float(matthews_corrcoef(labels, predicted)))


def compare_original(predictions, dataset_metrics, full_run):
    """Compare held-out scores and metrics with the published original results."""
    tolerance = 1e-12
    keys = ['dataset_id', 'fold_id', 'RNA_ID']
    reference = read_table(ROOT / 'results/cross_validation/oof_predictions.tsv')
    tasks = predictions[['dataset_id', 'fold_id']].drop_duplicates()
    reference = reference.merge(tasks, on=['dataset_id', 'fold_id'], how='inner')
    actual = predictions.set_index(keys)
    reference = reference.set_index(keys)
    if (not actual.index.is_unique or not reference.index.is_unique
            or len(actual) != len(reference) or set(actual.index) != set(reference.index)):
        raise ValueError('Original and reproduced OOF memberships differ.')
    reference = reference.loc[actual.index]
    if not np.array_equal(actual.label, reference.label):
        raise ValueError('Original and reproduced OOF labels differ.')
    score_difference = float(np.max(np.abs(actual.probability - reference.probability)))
    metric_rows, metric_difference = [], 0.0
    if dataset_metrics:
        current = pd.DataFrame(dataset_metrics).set_index('dataset_id')
        original = read_table(ROOT / 'results/cross_validation/dataset_oof_metrics.tsv').set_index('dataset_id')
        if not original.index.is_unique:
            raise ValueError('Original dataset metrics have repeated dataset IDs.')
        original = original.loc[current.index]
        metric_difference = float(np.max(np.abs(current[METRICS].to_numpy() - original[METRICS].to_numpy())))
        for name in METRICS:
            mean, original_mean = float(current[name].mean()), float(original[name].mean())
            sd = float(current[name].std(ddof=1)) if len(current) > 1 else None
            original_sd = float(original[name].std(ddof=1)) if len(current) > 1 else None
            metric_rows.append(dict(metric=name, mean=mean, SD=sd, n_datasets=len(current),
                                    original_mean=original_mean, original_SD=original_sd,
                                    mean_difference=mean-original_mean,
                                    SD_difference=sd-original_sd if sd is not None else None))
    matches = score_difference <= tolerance and metric_difference <= tolerance
    return dict(status='MATCH' if matches else 'DIFFERENT',
                scope='all 50 models' if full_run else 'selected folds only',
                full_reproduction_matches=bool(full_run and matches),
                absolute_tolerance=tolerance, oof_predictions_compared=len(actual),
                oof_scores_exactly_equal=bool(np.array_equal(actual.probability, reference.probability)),
                max_oof_score_difference=score_difference,
                max_dataset_metric_difference=metric_difference, metrics=metric_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/training')
    parser.add_argument('--features', type=Path, help='Feature folder from extract_features.py; default: frozen original features')
    parser.add_argument('--datasets', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 6)))
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    if (args.threads < 1 or len(set(args.datasets)) != len(args.datasets)
            or len(set(args.folds)) != len(args.folds)
            or not set(args.datasets) <= set(range(1, 11))
            or not set(args.folds) <= set(range(1, 6))):
        parser.error('Use positive threads, unique datasets 1..10 and unique folds 1..5.')
    dest = args.output.resolve()
    protected = [ROOT / p for p in ['data', 'models', 'results']]
    if dest == ROOT or any(dest == p or p in dest.parents for p in protected):
        parser.error('Choose an output directory outside the distributed inputs.')
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        parser.error('Output already exists and is nonempty; choose a new directory.')
    cfg = json.loads((ROOT / 'models/lightgbm_config.json').read_text())
    _, x, frames = load_inputs(args.features)
    dest.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('llps_rna_training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(dest / 'training.log')
    logger.addHandler(handler)
    lgb.register_logger(logger)
    params = dict(cfg['params'], num_threads=args.threads)
    started, fold_reports, dataset_metrics = time.monotonic(), [], []
    predictions = []
    for d in args.datasets:
        md = frames[d-1]
        dataset_pred = []
        for fold in args.folds:
            tr, va = md.loc[md.fold_id.ne(fold)], md.loc[md.fold_id.eq(fold)]
            fit_start = time.monotonic()
            # Validation data never reaches Dataset/train or a fitted preprocessing step.
            booster = lgb.train(params, lgb.Dataset(np.array(x[tr.feature_row]), label=tr.label.to_numpy()),
                                num_boost_round=cfg['rounds'])
            scores = booster.predict(np.array(x[va.feature_row]), num_threads=args.threads)
            model_id = f'dataset_{d:02d}_fold_{fold}'
            folder = dest / 'models'
            folder.mkdir(exist_ok=True)
            booster.save_model(str(folder / f'{model_id}.txt'))
            pred = va.copy()
            pred['probability'] = scores
            pred['predicted_label'] = (scores >= .5).astype(int)
            dataset_pred.append(pred)
            report = dict(model_id=model_id, dataset_id=d, fold_id=fold, n_train=len(tr), n_validation=len(va),
                          iterations=booster.current_iteration(),
                          elapsed_seconds=time.monotonic()-fit_start,
                          train_ids=tr.RNA_ID.tolist(), validation_ids=va.RNA_ID.tolist())
            fold_reports.append(report)
            print(f'{model_id}: {len(tr)} train / {len(va)} held out; training complete', flush=True)
        pooled = pd.concat(dataset_pred, ignore_index=True)
        predictions.append(pooled)
        if set(args.folds) == set(range(1, 6)):
            dataset_metrics.append(dict(dataset_id=d, **metrics(pooled.label, pooled.probability)))
    pooled_predictions = pd.concat(predictions, ignore_index=True)
    pooled_predictions.to_csv(dest / 'oof_predictions.tsv', sep='\t', index=False)
    comparison = compare_original(pooled_predictions, dataset_metrics, len(fold_reports)==50)
    if dataset_metrics:
        values = pd.DataFrame(dataset_metrics)
        values.to_csv(dest / 'dataset_oof_metrics.tsv', sep='\t', index=False)
        pd.DataFrame(comparison['metrics']).to_csv(dest / 'summary.tsv', sep='\t', index=False)
    report = dict(completed=True, python=platform.python_version(),
                  packages={p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm']},
                  configuration=dict(cfg, params=params), datasets=args.datasets, folds=args.folds,
                  fitted_models=len(fold_reports), full_50_model_reproduction=len(fold_reports)==50,
                  feature_source='original frozen GPU features' if args.features is None else 'user-supplied recomputed features',
                  comparison_with_original=comparison,
                  elapsed_seconds=time.monotonic()-started, model_reports=fold_reports)
    (dest / 'training_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Comparison with original ({comparison["scope"]}): {comparison["status"]}', flush=True)
    print(f'OOF predictions compared: {comparison["oof_predictions_compared"]}; '
          f'maximum score difference: {comparison["max_oof_score_difference"]:.6g}; '
          f'tolerance: {comparison["absolute_tolerance"]:.0e}', flush=True)
    for row in comparison['metrics']:
        print(f'{row["metric"]:<10} original={row["original_mean"]:.6f} '
              f'reproduced={row["mean"]:.6f} difference={row["mean_difference"]:+.3g}', flush=True)
    print(f'Finished: {len(fold_reports)} models. Results: {dest}', flush=True)


if __name__ == '__main__':
    main()
