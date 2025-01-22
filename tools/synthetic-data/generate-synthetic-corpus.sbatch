#!/bin/bash
#SBATCH --job-name=synth-corpus
#SBATCH -o synth-corpus-output-%j.txt
#SBATCH -M ukko
#SBATCH -p gpu-oversub
#SBATCH --constraint=a100
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t4:00:00

# parameters
vocab=$1
lang=$2

echo "generating synthetic data for vocab $vocab in language $lang"

# setup Python environment and CUDA
module load CUDA
source /wrk-vakka/group/natlibfi-annif/vllm-cuda-venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

# change to the correct directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects/tools/synthetic-data

# run the generation script
time python vllm-synthetic-data-gen.py \
  ../../shared-task-datasets/GND/GND-Subjects-${vocab}_dnb-skos-with-en-labels.ttl \
  $lang \
  ../../shared-task-datasets/TIBKAT/$vocab-subjects/data/train/$lang.tsv.gz \
  ../../shared-task-datasets/TIBKAT/$vocab-subjects/data/train/new-synth-$lang.tsv.gz
