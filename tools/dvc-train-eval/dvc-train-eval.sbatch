#!/bin/bash
#SBATCH --job-name=dvc-train-eval
#SBATCH -o dvc-train-eval-output-%j.txt
#SBATCH -M ukko
#SBATCH --mem=32G
#SBATCH -t4:00:00

vocab=$1
backend=$2
lang=$3

echo "Params: $vocab-$backend-$lang"

module purge
module load Python/3.11.5-GCCcore-13.2.0

# activate python virtualenv with DVC and Annif
source /wrk-vakka/group/natlibfi-annif/git/Annif/venv-turso/bin/activate

# 4 secs sleep between DVC starts to try to avoid DVC lock contention
# see https://github.com/iterative/dvc/issues/755#issuecomment-1083079350
SLEEP_DURATION=$(( (SLURM_JOB_ID % 16) * 4 ))
echo "Sleeping $SLEEP_DURATION seconds..."
sleep $SLEEP_DURATION

# change to the correct directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects/

# run DVC to train and evaluate models
dvc repro eval@$vocab-$backend-$lang
