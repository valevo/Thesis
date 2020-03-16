#!/bin/bash
#SBATCH -N 1
#SBATCH -t 30:00:00
#SBATCH --mem=70G

module load pre2019
module load Python/3.6.1-intel-2016b 

echo "Stat job $PBS_JOBID started at `date`"


#mkdir "$TMPDIR"/ThesisII
#mkdir "$TMPDIR"/ThesisII/data
#cp -r $HOME/ThesisII/data/"$lang" "$TMPDIR"/ThesisII/data
#cp -r $HOME/ThesisII/filtering "$TMPDIR"/ThesisII
#cp -r $HOME/ThesisII/stats "$TMPDIR"/ThesisII


rsync -a $HOME/ThesisII "$TMPDIR"/ --exclude data --exclude .git
cd "$TMPDIR"/ThesisII


lang=VI

echo 
echo "language: $lang"


mkdir data
cp -r $HOME/ThesisII/data/"$lang"_pkl "$TMPDIR"/ThesisII/data/
cp -r $HOME/ThesisII/data/reader.py "$TMPDIR"/ThesisII/data/
cp -r $HOME/ThesisII/data/corpus.py "$TMPDIR"/ThesisII/data/


# 2 4 8 16 32 64 81
for h in 64 81; do

python3.6 SRF_main_parallelised.py --lang=$lang --n_tokens=1000000 --hist_len=$h

echo 
echo "done with hist_len $h at `date`"

cp -r $TMPDIR/ThesisII/results/$lang/SRF $HOME/ThesisII/results/$lang/

echo "and copied"
echo


done


# cp -r $TMPDIR/ThesisII/results/"$lang"/ $HOME/ThesisII/results


echo "Job $PBS_JOBID ended at `date`"