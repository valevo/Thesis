#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --mem=20G


module load Python/3.6.1-intel-2016b 

echo "Job $PBS_JOBID started at `date`"

cp -r $HOME/Thesis "$TMPDIR"

cd "$TMPDIR"/Thesis

echo "Thesis copied; current dir: `pwd`"


python3.6 parallelisation_def.py


cp -rf "$TMPDIR"/Thesis/test $HOME


echo "Job $PBS_JOBID ended at `date`"