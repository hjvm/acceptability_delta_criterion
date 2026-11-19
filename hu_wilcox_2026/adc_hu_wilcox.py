"""The Acceptability Delta Criterion (Vázquez Martínez 2021) on the LI data of Hu, Wilcox, Song, Mahowald & Levy
(2026, TACL, doi:10.1162/TACL.a.611).

Their Prediction 2 (model log-probability difference within a pair, correlated with the human acceptability
difference) is the delta correlation of the 2021 paper's Figure 1. This script computes it, and the ADC, for their
GPT-2 and Llama-3-70B (their released summed log-probabilities), BERT and RoBERTa (pseudo-log-likelihood, scored
as in the 2023 GenBench paper) and a public Wikipedia 5-gram (log-probability and SLOR).

Data: their LI stimuli as released (Sprouse et al. 2013 + Mahowald et al. 2016), human = their z-scored Likert
ratings. Their notebook drops pairs whose two sentences are identical, then keeps pairs with word edit distance
<= the 75th percentile (4 words): 1,883 pairs.

For each scorer: z-score over the sentences of the filtered set; per pair, the model difference
dz = z(good) - z(bad) and the human difference dh likewise. Then
  BLiMP criterion  = share of pairs with dz > 0
  ADC(delta)       = share of pairs with sign(dz) == sign(dh) and |dh - dz| < delta      (2021, eqs. 1-2)
  r                = Pearson correlation of dz and dh (unchanged by the z-scoring)

Usage: bash fetch_data.sh && python adc_hu_wilcox.py
Outputs: results/adc_hu_wilcox.csv, results/adc_hu_wilcox_delta_corr.csv, results/adc_hu_wilcox_heatmap.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA, SCORES, RESULTS = HERE / "data", HERE / "scores", HERE / "results"


def word_edit_distance(s1, s2):
    """Word-level Levenshtein distance after stripping a final period or question mark, as in their notebook."""
    a, b = str(s1).rstrip(".").rstrip("?").split(), str(s2).rstrip(".").rstrip("?").split()
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev = d[:]
        d[0] = i
        for j, y in enumerate(b, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (x != y))
    return d[-1]


def sentence_series(good, bad, gv, bv):
    s = dict(zip(good, np.asarray(gv, dtype=float)))
    s.update(zip(bad, np.asarray(bv, dtype=float)))
    return pd.Series(s)


def main():
    g = dict(np.load(DATA / "results/li/gpt2.npz", allow_pickle=True))
    l = dict(np.load(DATA / "results/li/Meta-Llama-3-70B.npz", allow_pickle=True))
    assert (g["good_sentence"] == l["good_sentence"]).all()
    df = pd.DataFrame({"good": g["good_sentence"].astype(str), "bad": g["bad_sentence"].astype(str),
                       "h_good": g["LS_grammatical"].astype(float), "h_bad": g["LS_ungrammatical"].astype(float)})

    scorers = {"GPT-2": sentence_series(df.good, df.bad, g["good_sum_logprob"], g["bad_sum_logprob"]),
               "Llama-3-70B": sentence_series(df.good, df.bad, l["good_sum_logprob"], l["bad_sum_logprob"])}
    pll = pd.read_csv(SCORES / "hu_li_pll_scores.csv.gz").pivot_table(index="sentence", columns="model", values="pseudoLogProb")
    for name, col in (("RoBERTa-base", "roberta-base"), ("RoBERTa-large", "roberta-large"),
                      ("BERT-base-cased", "bert-base-cased"), ("BERT-large-cased", "bert-large-cased")):
        scorers[name] = pll[col]
    ng = pd.read_csv(SCORES / "hu_li_5gram_wiki_scores.csv").set_index("sentence")
    scorers["5-gram-Wiki-LogProb"], scorers["5-gram-Wiki-SLOR"] = ng.logprob, ng.slor

    df = df[df.good != df.bad].copy()
    df["ed"] = [word_edit_distance(a, b) for a, b in zip(df.good, df.bad)]
    f = df[df.ed <= df.ed.describe()["75%"]].drop_duplicates(["good", "bad"]).reset_index(drop=True)
    print("pairs", len(f))

    sentences = pd.unique(pd.concat([f.good, f.bad]))
    human = pd.concat([f.set_index("good").h_good, f.set_index("bad").h_bad])
    human = human[~human.index.duplicated()].reindex(sentences)
    hz = (human - human.mean()) / human.std(ddof=0)
    dh = hz.reindex(f.good).values - hz.reindex(f.bad).values

    deltas, rows = pd.DataFrame({"Human": f.h_good - f.h_bad}), []
    for name, s in scorers.items():
        x = s.reindex(sentences).astype(float)
        z = (x - x.mean()) / x.std(ddof=0)
        dz = z.reindex(f.good).values - z.reindex(f.bad).values
        deltas[name] = dz
        row = dict(model=name, blimp_criterion=float((dz > 0).mean()))
        for delta in (0.5, 1.0):
            row[f"adc_{delta}"] = float(((np.sign(dh) == np.sign(dz)) & (np.abs(dh - dz) < delta)).mean())
        row["r_with_human"] = float(np.corrcoef(dh, dz)[0, 1])
        rows.append(row)
    res = pd.DataFrame(rows).round(3)
    print(res.to_string())
    res.to_csv(RESULTS / "adc_hu_wilcox.csv", index=False)
    corr = deltas.corr()
    corr.to_csv(RESULTS / "adc_hu_wilcox_delta_corr.csv")

    c = corr.iloc[1:, :-1]
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sns.heatmap(c, mask=mask, annot=True, fmt=".2f", cmap=sns.light_palette("#c0392b", as_cmap=True), vmin=0, vmax=1,
                cbar=False, square=True, linewidths=.6, linecolor="white", annot_kws={"size": 10}, ax=ax)
    ax.set_title("Human and Model\nAcceptability Delta\nCorrelation Matrix", loc="right", fontsize=14, y=0.72, x=0.99)
    ax.tick_params(labelsize=9, length=0)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "adc_hu_wilcox_heatmap.png", dpi=250)


if __name__ == "__main__":
    main()
