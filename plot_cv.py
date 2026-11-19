#!/usr/bin/env python3
"""Plot ten frozen OOF ROC/PR curves and report their dataset-level metrics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'output/cv_figures')
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    df=pd.read_csv(root/'results/cross_validation/oof_predictions.tsv',sep='\t')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for d,rows in df.groupby('dataset_id',sort=True):
        fpr,tpr,_=roc_curve(rows.label,rows.probability)
        precision,recall,_=precision_recall_curve(rows.label,rows.probability)
        axes[0].plot(fpr,tpr,lw=1,label=f'D{d}: {roc_auc_score(rows.label,rows.probability):.3f}')
        axes[1].step(recall,precision,where='post',lw=1,label=f'D{d}: {average_precision_score(rows.label,rows.probability):.3f}')
    axes[0].plot([0,1],[0,1],'--',color='grey',lw=.8)
    axes[1].axhline(.5,ls='--',color='grey',lw=.8)
    for ax in axes:
        ax.set(xlim=(0,1),ylim=(0,1.01))
        ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=7,loc='lower right' if ax is axes[0] else 'lower left')
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='OOF ROC (AUROC)')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='OOF PR (average precision)')
    fig.tight_layout()
    a.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output/'cross_validation.pdf')
    fig.savefig(a.output/'cross_validation.png',dpi=300)
    plt.close(fig)


if __name__=='__main__':
    main()
