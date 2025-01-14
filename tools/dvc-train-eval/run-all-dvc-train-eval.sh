#!/bin/bash

for vocab in all tib-core; do
  for lang in de en; do
    sbatch dvc-train-eval.sbatch $vocab $lang
  done
done

