#!/bin/bash
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --mem=70G


module load Python/3.6.1-intel-2016b 

echo "Stat job $PBS_JOBID started at `date`"

mkdir "$TMPDIR"/Thesis

cp -r $HOME/Thesis/data "$TMPDIR"/Thesis
cp -r $HOME/Thesis/stats "$TMPDIR"/Thesis
cp -r $HOME/Thesis/Results "$TMPDIR"/Thesis
cp -r $HOME/Thesis/utils "$TMPDIR"/Thesis
cp -r $HOME/Thesis/stats_main.py "$TMPDIR"/Thesis


cd "$TMPDIR"/Thesis

echo "Thesis copied; current dir: `pwd`"


#ID KO NO TR VI
langs=(EO FI)
for i in "${!langs[@]}"; do
    echo "RUNNING STAT SCRIPT WITH ${langs[$i]}"
    python3.6 stats_main.py --lang=${langs[$i]} &
done

echo "WAITING..."

wait

for i in "${!langs[@]}"; do
    echo "COPYING ${langs[$i]}"
    cp -rf "$TMPDIR"/Thesis/Results/${langs[$i]} $HOME
done


echo "Job $PBS_JOBID ended at `date`"