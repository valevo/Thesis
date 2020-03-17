# Statistical Methodology for Quantitative Linguistics: <br> A Case Study of Learnability and Zipf’s Law 
__Master of Logic Thesis of Valentin Vogelmann__

## Contents

### `/figures`

Figures, tables and results, see `figures/README.md` for the list of contents.

### `/data`

Linguistically pre-processed (see Section 2.1 for pipeline details) and subsequently pickled (using Python module `pickle`) corpora: Wikipedia dumps in 7 languages. Each folder, prefixed with a language code (see below), contains the corpus split into multiple files in order to stay below GitHub's file size limit. Use `wiki_from_pickles` in `/data/reader.py`to load a corpus from a folder in `/data`; `corpus.py` contains wrappers to turn the corpora loaded with `wiki_from_pickles` into Python objects with convenient functionality.

Language codes are : Esperanto - `EO`, Finnish - `FI`, Indonesian - `ID`, Korean - `KO`, Norwegian (the Bokmal variant) - `NO`, Turkish - `TR` and Vietnamese - `VI`.

### `/src`

Code used for generating subcorpora according to the Subsampling and Filtering methods and analysing those.

 - `/src/filtering`: 
 - `/src/jackknife`:
 - `/src/evaluation`:
 - `shell_scripts`: shell scripts to deploy Python code on SurfSARA's LISA computing cluster

