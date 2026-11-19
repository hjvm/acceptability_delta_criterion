# The ADC on the Hu, Wilcox et al. (2026) data

Hu, Wilcox, Song, Mahowald & Levy (2026, TACL, [doi:10.1162/TACL.a.611](https://doi.org/10.1162/TACL.a.611))
test, as their Prediction 2, whether a model's log-probability difference within a minimal pair correlates with
the human acceptability difference. That is the delta correlation of this repository's 2021 paper (Figure 1).
This folder applies the 2021 analysis, the delta correlation and the Acceptability Delta Criterion, to their
LI data as released, next to their own GPT-2 and Llama-3-70B scores.

```bash
pip install pandas numpy matplotlib seaborn
bash hu_wilcox_2026/fetch_data.sh        # their released LI results (git-ignored, no redistribution licence)
python hu_wilcox_2026/adc_hu_wilcox.py
```

## Data and scoring

* **Pairs.** Their LI set (Sprouse et al. 2013 plus Mahowald et al. 2016), filtered as in their notebook: identical
  pairs dropped, then word edit distance at most 4 (their 75th percentile). 1,883 pairs.
* **Humans.** Their z-scored Likert ratings.
* **GPT-2, Llama-3-70B.** Their released summed log-probabilities.
* **BERT, RoBERTa.** Pseudo-log-likelihood, scored on CPU exactly as in the 2023 GenBench paper
  (`scores/hu_li_pll_scores.csv.gz`; the scoring script is `analyses_2026/score_pll.py` in
  [benchmarking_acquisition](https://github.com/hjvm/benchmarking_acquisition)).
* **5-gram.** The public Wikipedia KenLM 5-gram
  ([BramVanroy/kenlm_wikipedia_en](https://huggingface.co/BramVanroy/kenlm_wikipedia_en)), log-probability and SLOR
  (`scores/hu_li_5gram_wiki_scores.csv`). Not the BNC trigram of the 2021 paper, which covers only the Sprouse sentences.
* Each scorer is z-scored over the sentences of the filtered set; ADC as in the 2021 paper, equations 1 and 2.

## Results (`results/adc_hu_wilcox.csv`)

| Scorer | BLiMP criterion | ADC, δ = 0.5 | ADC, δ = 1.0 | r with human Δ |
|---|---|---|---|---|
| GPT-2 | .785 | .199 | .356 | .344 |
| Llama-3-70B | .845 | .245 | .419 | .399 |
| RoBERTa-base | .857 | .306 | .535 | .471 |
| RoBERTa-large | .868 | .320 | .551 | .466 |
| BERT-base-cased | .848 | .288 | .479 | .456 |
| BERT-large-cased | .851 | .280 | .494 | .473 |
| 5-gram, log-probability | .680 | .173 | .311 | .203 |
| 5-gram, SLOR | .727 | .228 | .393 | .351 |

* The GPT-2 and Llama-3 correlations are their published Prediction 2a values (.34 and .40). In the 2021 paper on
  LI-Adger, BERT reached .38 to .39 and the BNC trigram with SLOR .33.
* Every model drops from about 80% under the BLiMP criterion to 20 to 32% under the ADC.
* A 5-gram with SLOR matches GPT-2 as they scored it (.351 vs .344; ADC .228 vs .199).
* **Fairness.** GPT-2 and Llama-3 are shown with summed log-probability, the measure Hu, Wilcox et al. use for this
  test. With SLOR they reach .51 and .54 correlation. The 5-gram's raw log-probability gives .20.
* In the delta correlation matrix (`results/adc_hu_wilcox_heatmap.png`), GPT-2's differences correlate at .73 with
  the 5-gram's log-probability differences, more than with RoBERTa (.54) or BERT (.57).
