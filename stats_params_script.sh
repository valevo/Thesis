#!/bin/bash
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH --mem=40G


module load Python/3.6.1-intel-2016b 

echo "Stat params estimation job $PBS_JOBID started at `date`"

# cp -r $HOME/Thesis "$TMPDIR"


mkdir "$TMPDIR"/Thesis

cp -r $HOME/Thesis/stats "$TMPDIR"/Thesis
cp -r $HOME/Thesis/Results "$TMPDIR"/Thesis
cp -r $HOME/Thesis/utils "$TMPDIR"/Thesis
cp -r $HOME/Thesis/stats_params.py "$TMPDIR"/Thesis



cd "$TMPDIR"/Thesis

echo "Thesis copied; stats copied; current dir: `pwd`"
echo $(ls)


#langs=(EO  FI  ID  KO  NO  TR  VI)
#ns=(38269026 50000062 50000034 50000028 50003875 50000008 50000016)

langs=(EO  FI)
ns=(38269026 50000062)

for i in "${!langs[@]}"; do 
    echo "$i"
    echo "${langs[$i]}"
    echo "${ns[$i]}"
    
    cp -r $HOME/ThesisEstimationResults/$l/stats "$TMPDIR"/Thesis/Results/$l/
    echo "Copied stats folder"
    
    python3.6 stats_params.py --lang=${langs[$i]} --n=${ns[$i]} &
    
    echo 
done

wait

for i in "${!langs[@]}"; do 
    echo "$i"
    echo "${langs[$i]}"
    echo "${ns[$i]}"
    cp -r "$TMPDIR"/Thesis/Results/${langs[$i]}/stats/params $HOME/
done




echo "Job $PBS_JOBID ended at `date`"

