#!/bin/bash

for vocab in all tib-core; do
  for lang in de en; do
    # CPU stages
    #ANNIF_EVAL_JOBS=32 sbatch -p short -c 32 dvc-train-eval.sbatch $vocab mllm $lang
    #ANNIF_EVAL_JOBS=32 sbatch -p short -c 32 dvc-train-eval.sbatch $vocab bonsai $lang
    #ANNIF_EVAL_JOBS=32 sbatch -p short -c 32 dvc-train-eval.sbatch $vocab bm-ensemble $lang
    #ANNIF_EVAL_JOBS=32 sbatch -p short -c 32 dvc-train-eval.sbatch $vocab bmx-ensemble $lang
    # GPU stages
    #ANNIF_EVAL_JOBS=1 sbatch -p gpu -G 1 -c 32 -t4:00:00 dvc-train-eval.sbatch $vocab xtransformer $lang
    ANNIF_EVAL_JOBS=1 sbatch -p gpu -G 1 -c 8 -t4:00:00 dvc-train-eval.sbatch $vocab bmx-ensemble $lang
  done
done

