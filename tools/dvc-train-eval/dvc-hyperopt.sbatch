#!/bin/bash
#SBATCH --job-name=dvc-hyperopt
#SBATCH -o dvc-hyperopt-output-%j.txt
#SBATCH -M ukko
#SBATCH --mem=32G
#SBATCH -t4:00:00

vocab=$1
sources=$2
lang=$3

echo "Params: $vocab $sources $lang"

module purge
module load Python/3.11.5-GCCcore-13.2.0 CUDA

# activate python virtualenv with DVC and Annif
source /wrk-vakka/group/natlibfi-annif/git/Annif/venv-turso/bin/activate

# random sleep to try to avoid DVC lock contention
# see https://github.com/iterative/dvc/issues/755#issuecomment-1083079350
echo "Sleeping up to 20 seconds..."
sleep $((RANDOM % 20))

# change to the correct directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects/
 
# run DVC hyperparameter optimization
dvc repro hyperopt-ensemble@$vocab-$sources-$lang
