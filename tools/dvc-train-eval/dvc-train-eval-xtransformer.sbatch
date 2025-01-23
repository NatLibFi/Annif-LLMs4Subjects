#!/bin/bash
#SBATCH --job-name=dvc-train-eval-xtransformer
#SBATCH -o dvc-train-eval-xtransformer-output-%j.txt
#SBATCH -p gpu-oversub
#SBATCH -M ukko
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t3:00:00

vocab=$1
lang=$2

echo "Params: $vocab-$lang"

module purge
module load Python/3.11.5-GCCcore-13.2.0

# activate python virtualenv with DVC and Annif
source /wrk-vakka/group/natlibfi-annif/git/Annif/venv-turso/bin/activate

# random sleep to try to avoid DVC lock contention
# see https://github.com/iterative/dvc/issues/755#issuecomment-1083079350
echo "Sleeping up to 20 seconds..."
sleep $((RANDOM % 20))

# change to the correct directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects/

# run DVC to train and evaluate models
dvc repro xtransformer/dvc.yaml:eval@$vocab-xtransformer-$lang
