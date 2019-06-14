#!/bin/bash
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH --mem=40G


module load Python/3.6.1-intel-2016b 

echo "Stat summary job $PBS_JOBID started at `date`"

# cp -r $HOME/Thesis "$TMPDIR"


mkdir "$TMPDIR"/Thesis

cp -r $HOME/Thesis/stats "$TMPDIR"/Thesis
cp -r $HOME/Thesis/Results "$TMPDIR"/Thesis
cp -r $HOME/Thesis/utils "$TMPDIR"/Thesis
cp -r $HOME/Thesis/stats_eval.py "$TMPDIR"/Thesis



cd "$TMPDIR"/Thesis

echo "Thesis copied; stats copied; current dir: `pwd`"


langs=(EO  FI  ID  KO  NO  TR  VI)
ns=(38269026 50000062 50000034 50000028 50003875 50000008 50000016)

#langs=(EO TR)
#ns=(38269026 50000008)

for i in "${!langs[@]}"; do 
    echo "$i"
    echo "${langs[$i]}"
    echo "${ns[$i]}"
    
    cp -r $HOME/ThesisEstimationResults/${langs[$i]}/stats "$TMPDIR"/Thesis/Results/${langs[$i]}/
    echo "Copied stats folder"
    
    python3.6 stats_eval.py --lang=${langs[$i]} --n=${ns[$i]} 
    
    echo 
done


for i in "${!langs[@]}"; do 
    echo "$i"
    echo "${langs[$i]}"
    echo "${ns[$i]}"
    mkdir $HOME/${langs[$i]}
    cp -r "$TMPDIR"/Thesis/Results/${langs[$i]}/stats/summary $HOME/${langs[$i]}/
done




echo "Job $PBS_JOBID ended at `date`"

