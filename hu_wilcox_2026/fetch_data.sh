#!/usr/bin/env bash
# Downloads the Hu, Wilcox et al. (2026) LI results (their GPT-2 and Llama-3-70B scores, with the stimuli and
# human ratings) into hu_wilcox_2026/data/ (git-ignored). Source: github.com/jennhu/probability-grammaticality
set -euo pipefail
cd "$(dirname "$0")"
HU=https://raw.githubusercontent.com/jennhu/probability-grammaticality/master
for p in results/li/gpt2.npz results/li/Meta-Llama-3-70B.npz; do
  mkdir -p "data/$(dirname "$p")"; curl -fsSL "$HU/$p" -o "data/$p"; echo "  $p"
done
