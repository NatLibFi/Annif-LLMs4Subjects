#!/bin/bash
#SBATCH --job-name=dvc-hyperopt
#SBATCH -o dvc-hyperopt-output-%j.txt
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem=16G
#SBATCH -t4:00:00

vocab=$1
lang=$2

echo "Params: $vocab $lang"

# activate python virtualenv with DVC and Annif
source /wrk-vakka/group/natlibfi-annif/git/Annif/venv/bin/activate

# random sleep to try to avoid DVC lock contention
# see https://github.com/iterative/dvc/issues/755#issuecomment-1083079350
echo "Sleeping up to 20 seconds..."
sleep $((RANDOM % 20))

# change to the correct directory
cd /wrk-vakka/group/natlibfi-annif/git/Annif-LLMs4Subjects/
 
# run DVC hyperparameter optimization
dvc repro hyperopt-ensemble@$vocab-$lang
