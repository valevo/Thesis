#!/bin/bash
#SBATCH -N 1
#SBATCH -t 15:00:00
#SBATCH --mem=40G


module load Python/3.6.1-intel-2016b 

l="EO"
n=38269026


echo "Stat job $PBS_JOBID with ($l, $n) started at `date`"


cp -r $HOME/Thesis "$TMPDIR"

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
    
    cp -r $HOME/ThesisEstimationResults/$l/stats "$TMPDIR"/Thesis/$l/
    echo "Copied stats folder"
    
    python3.6 stats_params.py --lang=${langs[$i]} --n=${ns[$i]} &
done

wait

for i in "${!langs[@]}"; do 
    echo "$i"
    echo "${langs[$i]}"
    echo "${ns[$i]}"
    cp -r "$TMPDIR"/Thesis/Results/${langs[$i]}/stats/params $HOME/
done




echo "Job $PBS_JOBID ended at `date`"

