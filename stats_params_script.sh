#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --mem=10G


module load Python/3.6.1-intel-2016b 

l="EO"
n=38269026


echo "Stat job $PBS_JOBID with ($l, $n) started at `date`"


cp -r $HOME/Thesis "$TMPDIR"

cp -r $HOME/ThesisEstimationResults/$l/stats "$TMPDIR"/Thesis/$l

cd "$TMPDIR"/Thesis

echo "Thesis copied; stats copied; current dir: `pwd`"
echo $(ls)

# python3.6 stats_params.py --lang=$l --n=$n
mkdir Results/$l/stats/params
touch Results/$l/stats/params/test_file


cp -r Results/$l/stats/params $HOME/

echo "Job $PBS_JOBID ended at `date`"

