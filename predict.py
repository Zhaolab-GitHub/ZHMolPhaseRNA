#!/usr/bin/env python3
"""RNA sequence prediction: ERNIE768 + Handcrafted131 + 50 LightGBM models.

Use the command-line interface or import Predictor for repeated predictions.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import threading
import zlib

import numpy as np

from extract_features import FEATURE_COLUMNS, vector

DEFAULT_MODELS = Path(__file__).resolve().parent / 'models'


def normalize_sequence(sequence):
    if not isinstance(sequence, str):
        raise ValueError('RNA sequence must be a string.')
    sequence = ''.join(sequence.split()).upper().replace('T', 'U')
    if not sequence:
        raise ValueError('RNA sequence is empty.')
    invalid = sorted(set(sequence) - set('ACGU'))
    if invalid:
        raise ValueError('Only A/C/G/U (or T) are supported; ambiguous bases are not imputed: '
                         + ', '.join(invalid))
    return sequence


class Predictor:
    """Load once per process, reuse for every request. Calls are serialized.

    Torch thread count is process-global. Use a single Predictor per process.
    The server and CLI both use CPU float32 and 1000-nt nonoverlapping windows.
    """

    def __init__(self, models_dir=None, threads=4):
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError('threads must be a positive integer.')
        self.threads = threads
        self.models_dir = Path(models_dir or DEFAULT_MODELS).resolve()
        self._lock = threading.Lock()
        self.manifest = json.loads((self.models_dir / 'manifest.json').read_text())
        manifest = self.manifest
        expected_pairs = [(d, f) for d in range(1, 11) for f in range(1, 6)]
        if (manifest['format_version'] != 1 or manifest['feature_dim'] != 899
                or manifest['protocol']['window'] != 1000 or manifest['protocol']['stride'] != 1000
                or manifest['feature_columns'][768:] != list(FEATURE_COLUMNS)
                or len(manifest['feature_columns']) != 899
                or [(m['dataset_id'], m['fold_id']) for m in manifest['models']] != expected_pairs
                or len({m['model_id'] for m in manifest['models']}) != 50):
            raise ValueError('Expected the frozen ERNIE768 + Handcrafted131, 10 datasets x 5 folds manifest.')
        # gzip length is one trained feature; changing Python/zlib may change it.
        if sys.version_info[:2] != (3, 10) or zlib.ZLIB_RUNTIME_VERSION != '1.3.1':
            raise RuntimeError('Use Python 3.10 and zlib 1.3.1 from environment.yml '
                               'to preserve the trained gzip feature.')
        if not (self.models_dir / 'ernie_rna/pytorch_model.bin').is_file():
            raise ValueError('Download ERNIE-RNA pytorch_model.bin into models/ernie_rna/ '
                             'as described in README.md, section 3.')
        self._load_models()

    def _load_models(self):
        import lightgbm as lgb
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(42)
        checkpoint = self.models_dir / 'ernie_rna'
        # The supplied HF port includes a fallback absolute config import.
        sys.path.insert(0, str(checkpoint))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint), local_files_only=True, trust_remote_code=True, use_fast=False)
            self.encoder, loading = AutoModel.from_pretrained(
                str(checkpoint), local_files_only=True, trust_remote_code=True,
                attn_implementation='eager', output_loading_info=True)
        finally:
            sys.path.remove(str(checkpoint))
        if (loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs')
                or any(not k.startswith('lm_head.') for k in loading.get('unexpected_keys', []))):
            raise ValueError('Incomplete or incompatible ERNIE-RNA checkpoint: ' + str(loading))
        if (self.encoder.config.embed_dim != 768 or self.encoder.config.num_layers != 12
                or self.encoder.config.max_positions < 1002):
            raise ValueError('Unexpected ERNIE-RNA architecture.')
        self.encoder.eval().requires_grad_(False).to(device='cpu', dtype=torch.float32)
        self.models = [lgb.Booster(model_file=str(self.models_dir / m['path']))
                       for m in self.manifest['models']]
        if any(m.num_feature() != 899 or m.current_iteration() != 800 for m in self.models):
            raise ValueError('Expected 50 LightGBM boosters with 899 features and 800 iterations.')

    def _embedding(self, sequence):
        total = np.zeros(768, dtype=np.float64)
        # Same starts as original window_starts(L, 1000, 1000), including the
        # short final window. Weight by base count, never average window means.
        for start in range(0, len(sequence), 1000):
            window = sequence[start:start + 1000]
            encoded = self.tokenizer(window, padding=True, truncation=False,
                                     add_special_tokens=True, return_tensors='pt')
            ids = encoded['input_ids'][0]
            if (len(ids) != len(window) + 2 or int(ids[0]) != self.tokenizer.cls_token_id
                    or int(ids[-1]) != self.tokenizer.eos_token_id
                    or int(encoded['attention_mask'][0].sum()) != len(window) + 2):
                raise ValueError('Tokenization must be one nucleotide per token plus CLS/EOS.')
            with self.torch.inference_mode():
                hidden = self.encoder(**{k: v.to('cpu') for k, v in encoded.items()},
                                      output_hidden_states=False, output_attentions=False,
                                      return_dict=True).last_hidden_state
                bases = hidden[0, 1:1 + len(window)].detach().float().cpu().numpy()
            if bases.shape != (len(window), 768) or not np.isfinite(bases).all():
                raise ValueError('Invalid ERNIE-RNA nucleotide representations.')
            total += bases.astype(np.float64).sum(axis=0)
        return (total / len(sequence)).astype(np.float32)

    def features(self, sequence):
        """Return a single (899,) float32 vector in frozen training column order."""
        sequence = normalize_sequence(sequence)
        with self._lock:
            return self._features(sequence)

    def _features(self, sequence):
        # FastAPI executes sync handlers on worker threads. OpenMP's thread
        # setting may differ there from the thread that loaded the encoder.
        if self.torch.get_num_threads() != self.threads:
            self.torch.set_num_threads(self.threads)
        x = np.concatenate((self._embedding(sequence), vector(sequence)))
        if x.shape != (899,) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('Invalid fused feature vector.')
        return x

    def predict(self, sequence, rna_id='query', include_model_scores=False):
        sequence = normalize_sequence(sequence)
        with self._lock:
            x = self._features(sequence)
            scores = np.array([float(m.predict(x[None, :], num_threads=self.threads)[0])
                               for m in self.models], dtype=np.float64)
        if scores.shape != (50,) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('Invalid model probabilities.')
        result = dict(RNA_ID=rna_id, length=len(sequence),
                      n_models=50, mean_probability=float(scores.mean()),
                      SD_between_50_models=float(scores.std(ddof=1)),
                      median_probability=float(np.median(scores)),
                      min_probability=float(scores.min()), max_probability=float(scores.max()),
                      positive_vote_fraction=float((scores >= .5).mean()),
                      predicted_label_at_mean_0p5=int(scores.mean() >= .5))
        if include_model_scores:
            result['model_probabilities'] = {m['model_id']: float(p)
                                           for m, p in zip(self.manifest['models'], scores)}
        return result


def fasta_records(path):
    header, parts = None, []
    seen = set()
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, normalize_sequence(''.join(parts))
                text = line[1:].strip()
                if not text:
                    raise ValueError('Empty FASTA header.')
                header = text.split('|', 1)[0].split()[0]
                if header in seen:
                    raise ValueError('Duplicate FASTA ID: ' + header)
                seen.add(header)
                parts = []
            elif header is None:
                raise ValueError('FASTA sequence appears before its header.')
            else:
                parts.append(line)
    if header is None:
        raise ValueError('Empty FASTA.')
    yield header, normalize_sequence(''.join(parts))


def main():
    p = argparse.ArgumentParser(description='CPU: RNA -> ERNIE768 + Handcrafted131 -> mean of 50 models')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--sequence', help='One RNA sequence; whitespace/lowercase/T accepted')
    source.add_argument('--fasta', type=Path, help='FASTA; invalid input fails explicitly')
    p.add_argument('--id', default='query', help='ID for --sequence')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--models-dir', type=Path)
    p.add_argument('--output', type=Path, help='Default: stdout')
    p.add_argument('--format', choices=['json', 'tsv'], default='json')
    p.add_argument('--include-model-scores', action='store_true')
    a = p.parse_args()
    try:
        # Validate all records before allocating models or opening output files.
        records = ([(a.id, normalize_sequence(a.sequence))] if a.sequence is not None
                   else list(fasta_records(a.fasta)))
        predictor = Predictor(a.models_dir, a.threads)
        results = [predictor.predict(s, rid, a.include_model_scores) for rid, s in records]
    except (ValueError, RuntimeError, OSError) as error:
        p.exit(2, 'Error: ' + str(error) + '\n')
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
    output = a.output.open('w') if a.output else sys.stdout
    try:
        if a.format == 'json':
            json.dump(results[0] if a.sequence is not None else results, output,
                      ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
        else:
            rows = []
            for result in results:
                row = dict(result)
                scores = row.pop('model_probabilities', {})
                row.update(scores)
                rows.append(row)
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
    finally:
        if a.output:
            output.close()


if __name__ == '__main__':
    main()
