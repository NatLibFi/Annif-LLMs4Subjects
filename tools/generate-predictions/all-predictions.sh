#!/bin/bash

subset=$1  # dev or test
backend=$2 # e.g. bm-ensemble

for vocab in all tib-core; do
  for lang in de en de,en; do

    sbatch -p short -c 2 generate-predictions.sbatch gnd-$vocab-$backend-LANG $lang \
      $subset-$vocab-$backend-$lang.zip \
      ../../shared-task-datasets/TIBKAT/$vocab-subjects/data/$subset-*.jsonl.gz
  done
done
