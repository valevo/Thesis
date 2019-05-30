#!/bin/bash
#SBATCH -N 1
#SBATCH -t 07:00:00
#SBATCH --mem=70G


module load Python/3.6.1-intel-2016b 

echo "Stat job $PBS_JOBID started at `date`"

cp -r $HOME/Thesis "$TMPDIR"

cd "$TMPDIR"/Thesis

echo "Thesis copied; current dir: `pwd`"


#ID KO NO TR VI
for l in EO FI; do
    echo "RUNNING STAT SCRIPT WITH $l"
    python3.6 stats_main.py --lang=$l &
done

echo "WAITING..."

wait

for l in EO FI; do
    echo "COPYING $l"
    cp -rf "$TMPDIR"/Thesis/Results/$l $HOME
done


echo "Job $PBS_JOBID ended at `date`"