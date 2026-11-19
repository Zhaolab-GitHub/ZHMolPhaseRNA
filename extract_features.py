#!/usr/bin/env python3
"""Compute the fixed 131 sequence descriptors and extract 899 features from FASTA.

Pure sequence functions can be imported without loading the RNA encoder.
The command-line entry point loads the encoder only when main() is called.
"""
import argparse
from collections import Counter
import csv
import gzip
import itertools
import math
from pathlib import Path
import shutil

import numpy as np


BASES = ["A", "U", "G", "C"]
KMER_NAMES = {
    k: tuple("k{}_{}".format(k, "".join(parts)) for parts in itertools.product(BASES, repeat=k))
    for k in (2, 3)
}
BASE_LOOKUP = np.full(256, -1, dtype=np.int16)
for _index, _base in enumerate(BASES):
    BASE_LOOKUP[ord(_base)] = _index


def normalize_seq(seq):
    seq = "".join(seq.split()).upper().replace("T", "U")
    invalid = set(seq) - set("ACGUN")
    if invalid:
        raise ValueError("Non-nucleotide characters: {}".format(sorted(invalid)))
    return seq


def shannon_entropy(seq):
    n = len(seq)
    if n == 0:
        return 0.0

    counts = Counter(seq)
    ent = 0.0

    for b in BASES:
        p = counts.get(b, 0) / n
        if p > 0:
            ent -= p * math.log2(p)

    return ent


def longest_homopolymer(seq):
    if not seq:
        return 0

    values = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    boundaries = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
    return int(np.diff(boundaries).max())


def kmer_frequencies(seq, k):
    if k <= 0:
        raise ValueError("k must be positive")
    names = KMER_NAMES.get(k)
    if names is None:
        names = tuple("k{}_{}".format(k, "".join(parts)) for parts in itertools.product(BASES, repeat=k))
    size = len(seq) - k + 1
    if size <= 0:
        return dict.fromkeys(names, 0.0)
    codes = BASE_LOOKUP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    indices = np.zeros(size, dtype=np.int64)
    valid = np.ones(size, dtype=np.bool_)
    for offset in range(k):
        part = codes[offset:offset + size]
        valid &= part >= 0
        indices = indices * 4 + np.maximum(part, 0)
    counts = np.bincount(indices[valid], minlength=len(names)).astype(np.float64)
    total = counts.sum()
    if total:
        counts /= total
    return dict(zip(names, counts))


def _extract_original_features(seq):
    seq = normalize_seq(seq)
    n = len(seq)

    counts = Counter(seq)

    A = counts.get("A", 0)
    U = counts.get("U", 0)
    G = counts.get("G", 0)
    C = counts.get("C", 0)
    N = counts.get("N", 0)

    valid_len = A + U + G + C
    longest = longest_homopolymer(seq)

    def safe_div(x, y):
        return x / y if y > 0 else 0.0

    features = {
        "seq_len": n,
        "valid_len": valid_len,
        "n_count": N,
        "invalid_fraction": safe_div(n - valid_len, n),

        "A_frac": safe_div(A, valid_len),
        "U_frac": safe_div(U, valid_len),
        "G_frac": safe_div(G, valid_len),
        "C_frac": safe_div(C, valid_len),

        "GC_content": safe_div(G + C, valid_len),
        "AU_content": safe_div(A + U, valid_len),
        "purine_content": safe_div(A + G, valid_len),
        "pyrimidine_content": safe_div(C + U, valid_len),

        "G_minus_C": safe_div(G - C, valid_len),
        "A_minus_U": safe_div(A - U, valid_len),

        "entropy": shannon_entropy(seq),
        "longest_homopolymer": longest,
        "longest_homopolymer_frac": safe_div(longest, n),
    }

    features.update(kmer_frequencies(seq, k=2))
    features.update(kmer_frequencies(seq, k=3))

    return features


ORIGINAL_COLUMNS=tuple(_extract_original_features("ACGU"))
ADDED_COLUMNS=(
    'k2_entropy_norm','k3_entropy_norm','linguistic_complexity_k1_5','gzip_bytes_per_nt',
    *[f'local_entropy{w}_{stat}' for w in (12,30) for stat in ('mean','min','std','lowfrac')],
    *[f'homo_{b}_{stat}' for b in 'AUGC' for stat in ('maxrun','count_ge4_per100nt','coverage_ge4_frac')],
    *[f'tandem_{k}_{stat}' for k in (2,3) for stat in ('tracts_per100nt','coverage_frac','max_tract_frac')],
    *[f'motif_{m}_window30_{stat}' for m in ('UG','GU') for stat in ('max_density','density_std')],
)
FEATURE_COLUMNS=ORIGINAL_COLUMNS+ADDED_COLUMNS
assert len(ORIGINAL_COLUMNS)==97 and len(ADDED_COLUMNS)==34 and len(FEATURE_COLUMNS)==len(set(FEATURE_COLUMNS))==131


def normalized_sequence(seq):
    seq=normalize_seq(seq)
    if not seq:raise ValueError('Empty RNA is outside the feature definition.')
    if set(seq)-set('ACGU'):
        raise ValueError('The new 34 features require canonical A/C/G/U; ambiguous N is rejected, not imputed.')
    return seq


def kmer_entropy(seq,k):
    n=len(seq)-k+1
    if n<=0:return 0.
    counts=Counter(seq[i:i+k] for i in range(n))
    return -sum((c/n)*math.log2(c/n) for c in counts.values())/(2*k)


def linguistic_complexity(seq):
    ratios=[]
    for k in range(1,min(5,len(seq))+1):
        n=len(seq)-k+1
        ratios.append(len({seq[i:i+k] for i in range(n)})/min(4**k,n))
    return float(np.mean(ratios))


def local_entropies(seq,window):
    values=np.frombuffer(seq.encode('ascii'),dtype=np.uint8)
    width=min(window,len(values))
    counts=[]
    for base in b'AUGC':
        cumulative=np.r_[0,np.cumsum(values==base,dtype=np.int64)]
        counts.append(cumulative[width:]-cumulative[:-width])
    p=np.asarray(counts,dtype=np.float64)/width
    return -np.sum(p*np.log2(np.where(p>0,p,1.)),axis=0)/2.


def homopolymer_features(seq):
    values=np.frombuffer(seq.encode('ascii'),dtype=np.uint8)
    boundaries=np.flatnonzero(np.r_[True,values[1:]!=values[:-1],True])
    lengths=np.diff(boundaries);bases=values[boundaries[:-1]]
    output={}
    for base in 'AUGC':
        selected=lengths[bases==ord(base)];long=selected[selected>=4]
        output[f'homo_{base}_maxrun']=int(selected.max()) if len(selected) else 0
        output[f'homo_{base}_count_ge4_per100nt']=len(long)*100/len(seq)
        output[f'homo_{base}_coverage_ge4_frac']=int(long.sum())/len(seq)
    return output


def tandem_intervals(seq,k):
    """Maximal period-k runs; merge overlapping (not merely touching) intervals.

    k=2,3 are prime: a primitive motif has fundamental period k iff it is
    not a single repeated base. Partial motif tails are retained by maximal
    periodic intervals / union of shifted repeat detections.
    """
    if k not in (2,3):raise ValueError('Only fundamental periods 2 and 3 are defined.')
    if len(seq)<3*k:return []
    values=np.frombuffer(seq.encode('ascii'),dtype=np.uint8)
    equal=values[:-k]==values[k:]
    boundaries=np.flatnonzero(np.diff(np.r_[False,equal,False].astype(np.int8)))
    intervals=[]
    for a,b in boundaries.reshape(-1,2):
        if b-a<2*k or len(set(seq[a:a+k]))==1:continue
        lo,hi=int(a),int(b+k)
        if intervals and lo<intervals[-1][1]:
            intervals[-1]=(intervals[-1][0],max(hi,intervals[-1][1]))
        else:intervals.append((lo,hi))
    return intervals


def motif_densities(seq,motif,window=30):
    n=len(seq)
    if n<2:return np.zeros(1,dtype=np.float64)
    width=min(window,n);values=np.frombuffer(seq.encode('ascii'),dtype=np.uint8)
    mask=(values[:-1]==ord(motif[0]))&(values[1:]==ord(motif[1]))
    cumulative=np.r_[0,np.cumsum(mask,dtype=np.int64)]
    span=width-1
    return (cumulative[span:]-cumulative[:-span]).astype(np.float64)/max(span,1)


def extract_features_from_seq(seq):
    seq=normalized_sequence(seq)
    # Retain the original 97 formulas, insertion order, entropy and
    # valid-kmer denominators before adding the 34 descriptors.
    output=_extract_original_features(seq)
    output['k2_entropy_norm']=kmer_entropy(seq,2)
    output['k3_entropy_norm']=kmer_entropy(seq,3)
    output['linguistic_complexity_k1_5']=linguistic_complexity(seq)
    output['gzip_bytes_per_nt']=len(gzip.compress(seq.encode('ascii'),compresslevel=9,mtime=0))/len(seq)
    for width in (12,30):
        ent=local_entropies(seq,width)
        output[f'local_entropy{width}_mean']=float(ent.mean())
        output[f'local_entropy{width}_min']=float(ent.min())
        output[f'local_entropy{width}_std']=float(ent.std(ddof=0))
        output[f'local_entropy{width}_lowfrac']=float((ent<.75).mean())
    output.update(homopolymer_features(seq))
    for k in (2,3):
        intervals=tandem_intervals(seq,k);lengths=[b-a for a,b in intervals]
        output[f'tandem_{k}_tracts_per100nt']=len(lengths)*100/len(seq)
        output[f'tandem_{k}_coverage_frac']=sum(lengths)/len(seq)
        output[f'tandem_{k}_max_tract_frac']=max(lengths,default=0)/len(seq)
    for motif in ('UG','GU'):
        values=motif_densities(seq,motif)
        output[f'motif_{motif}_window30_max_density']=float(values.max())
        output[f'motif_{motif}_window30_density_std']=float(values.std(ddof=0))
    if tuple(output)!=FEATURE_COLUMNS or not np.isfinite(np.asarray(list(output.values()),dtype=np.float64)).all():
        raise ValueError('Wrong feature names/order/dimension or non-finite output.')
    return output


def vector(seq):
    result=extract_features_from_seq(seq)
    values=np.asarray([result[c] for c in FEATURE_COLUMNS],dtype=np.float32)
    if values.shape!=(131,) or not np.isfinite(values).all():raise ValueError('Invalid float32 131-vector.')
    return values


def feature_definitions():
    rows=[]
    for name in ORIGINAL_COLUMNS:
        rows.append(dict(name=name,category='original_97',definition='Original _extract_original_features formulas, denominators, units and order.',edge_case='Legacy behavior; extraction inputs are nonempty canonical ACGU.'))
    for name in ADDED_COLUMNS:
        if name in ('k2_entropy_norm','k3_entropy_norm'):
            definition='Overlapping k-mer Shannon entropy / (2*k); k='+name[1]
            edge='L<k:0; range[0,1].'
        elif name=='linguistic_complexity_k1_5':
            definition='Mean U_k/min(4^k,L-k+1) over k=1..5 with L>=k.';edge='Only valid k terms; range[0,1].'
        elif name=='gzip_bytes_per_nt':
            definition='len(gzip.compress(normalized_ASCII,compresslevel=9,mtime=0))/L.';edge='No clipping; gzip header makes short RNAs length-sensitive; record Python and zlib versions.'
        elif name.startswith('local_entropy'):
            definition='Step1 nt; mono-base Shannon entropy/2 in window12 or30; named mean/min/population std/fraction strictly<0.75.';edge='L<window:one whole-RNA window, std=0; range[0,1].'
        elif name.startswith('homo_'):
            definition='Named base: longest run; number of maximal runs>=4 *100/L; or sum length of such runs/L.';edge='Missing base:0. A long run counts once.'
        elif name.startswith('tandem_'):
            definition='Fundamental period k=2 or3; at least3 motif copies. Maximal periodic intervals, deduplicate, merge strict overlaps; count*100/L, union length/L, or longest merged tract/L.';edge='L<3k:0; homopolymers excluded; touching intervals stay separate; shifted partial motif tails retained in union.'
        else:
            definition='UG or GU overlapping occurrence count/max(actual_window_length-1,1) in step1 30-nt windows; named max or population std.';edge='L<30:whole-RNA window; L<2:0; one window std=0.'
        rows.append(dict(name=name,category='new_34',definition=definition,edge_case=edge))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fasta', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--models-dir', type=Path)
    a = p.parse_args()
    if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
        p.error('Output must be a new or empty directory.')
    # Import at runtime: predict imports the pure feature functions above.
    from predict import Predictor, fasta_records

    records = list(fasta_records(a.fasta))
    model = Predictor(a.models_dir, a.threads)
    a.output.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(a.output / 'X_899.npy', mode='w+', dtype=np.float32, shape=(len(records),899))
    # Completion metadata is written only after every feature row succeeds.
    rows = []
    for i,(rid,seq) in enumerate(records):
        x[i] = model.features(seq)
        rows.append(dict(row_idx=i,RNA_ID=rid,length=len(seq)))
        if i % 25 == 0 or i+1 == len(records):
            print(f'Features: {i+1}/{len(records)}', flush=True)
    x.flush()
    shutil.copyfile(model.models_dir / 'feature_columns_899.txt', a.output / 'feature_columns_899.txt')
    with (a.output / 'metadata.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
