#!/bin/bash
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --mem=25G

module load pre2019
module load Python/3.6.1-intel-2016b 

pip install --user lexical-diversity
pip install --user tag-fixer
python3 -c "from lexical_diversity import lex_div; print(lex_div)"

echo "evaluation_main job $PBS_JOBID started at `date`"

rsync -a $HOME/ThesisII "$TMPDIR"/ --exclude data --exclude .git

cd "$TMPDIR"/ThesisII

mkdir data
cp $HOME/ThesisII/data/reader.py "$TMPDIR"/ThesisII/data/
cp $HOME/ThesisII/data/corpus.py "$TMPDIR"/ThesisII/data/

lang=KO


echo "language: $lang"

cp -r $HOME/ThesisII/data/"$lang"_pkl "$TMPDIR"/ThesisII/data/


python3 typicality_eval.py --lang=$lang --factors 2 6 10 14 --hist_lens 2 4 8 16 32 64 81

echo
echo "done with typicality evaluation at `date`"
cp -r $TMPDIR/ThesisII/results/$lang/evaluation $HOME/ThesisII/results/$lang/
echo "and copied"
echo


python3 diversity_eval.py --lang=$lang --factors 2 6 10 14 --hist_lens 2 4 8 16 32 64 81

echo
echo "done with diversity evaluation at `date`"
cp -r $TMPDIR/ThesisII/results/$lang/evaluation $HOME/ThesisII/results/$lang/
echo "and copied"
echo


python3 normality_eval.py --lang=$lang --factors 2 6 10 14 --hist_lens 2 4 8 16 32 64 81

echo
echo "done with normality evaluation at `date`"
cp -r $TMPDIR/ThesisII/results/$lang/evaluation $HOME/ThesisII/results/$lang/
echo "and copied"
echo


echo "Job $PBS_JOBID ended at `date`"