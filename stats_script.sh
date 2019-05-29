#!/bin/bash
#SBATCH -N 1
#SBATCH -t 07:00:00
#SBATCH --mem=40G


module load Python/3.6.1-intel-2016b 

echo "Stat job $PBS_JOBID started at `date`"

cp -r $HOME/Thesis "$TMPDIR"

cd "$TMPDIR"/Thesis

echo "Thesis copied; current dir: `pwd`"

#NO TR VI

for l in EO FI ID KO; do
    echo "RUNNING STAT SCRIPT WITH $l"
    python3.6 stats_main.py --lang=$l &
done

echo "WAITING..."

wait

for l in EO FI ID KO; do
    echo "COPYING $l"
    cp -rf "$TMPDIR"/Thesis/Results/$l $HOME
done


echo "Job $PBS_JOBID ended at `date`"