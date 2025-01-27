#!/bin/bash

for vocab in all tib-core; do
  for lang in de en; do
    # CPU stages
    ANNIF_EVAL_JOBS=32 sbatch -p short -c 32 dvc-train-eval.sbatch $vocab mllm $lang
    ANNIF_EVAL_JOBS=32 sbatch -p short -c 32 dvc-train-eval.sbatch $vocab bonsai $lang
    # GPU stages
    ANNIF_EVAL_JOBS=1 sbatch -p gpu-oversub -c 8 -t4:00:00 dvc-train-eval.sbatch $vocab xtransformer $lang
    #ANNIF_EVAL_JOBS=1 sbatch -p gpu -c 32 dvc-train-eval.sbatch $vocab ensemble $lang
  done
done

