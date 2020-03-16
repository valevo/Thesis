#!/bin/bash
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --mem=30G

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


mkdir data

cp -r $HOME/ThesisII/data/reader.py "$TMPDIR"/ThesisII/data/
cp -r $HOME/ThesisII/data/corpus.py "$TMPDIR"/ThesisII/data/

for lang in EO FI ID KO NO TR VI; do
    echo "language: $lang"
    cp -r $HOME/ThesisII/data/"$lang"_pkl "$TMPDIR"/ThesisII/data/

    python3.6 UNI_main.py --lang=$lang --n_tokens=1000000

    cp -r $TMPDIR/ThesisII/results/$lang/UNI $HOME/ThesisII/results/$lang/

    echo "... and copied"
    echo


done


# cp -r $TMPDIR/ThesisII/results/"$lang"/ $HOME/ThesisII/results


echo "Job $PBS_JOBID ended at `date`"