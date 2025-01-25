#!/bin/bash

for vocab in all tib-core; do
  for lang in de en; do
    # CPU stages
    sbatch -p short -c 32 dvc-train-eval.sbatch $vocab $lang eval@$vocab-mllm-$lang
    sbatch -p short -c 32 dvc-train-eval.sbatch $vocab $lang eval@$vocab-bonsai-$lang
    # GPU stages
    sbatch -p gpu -c 32 -t8:00:00 dvc-train-eval.sbatch $vocab $lang eval@$vocab-xtransformer-$lang
    # sbatch -p gpu -c 32 dvc-train-eval.sbatch $vocab $lang eval@$vocab-ensemble-$lang
  done
done

