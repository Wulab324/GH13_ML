# GH13_ML
## Decipher the mechanism of transglycosylation and hydrolysis in a GH13 family using machine learning 
----------------
Reference source 

Gado, J.E., Harrison, B.E., Sandgren, M., Ståhlberg, J., Beckham, G.T., and Payne, C.M. **Machine learning reveals sequence-function relationships in family 7 glycoside hydrolases.** Submitted to *FEBS* (2020).
https://github.com/jafetgado/Cel7ML

## Python version 
-----------------
- Python (3.6.8)

##### Python modules (version used in this work)
- pandas (1.1.1)
- numpy (1.19.1)
- scipy (1.5.2)
- biopython (1.77)
- scikit-learn (0.23.2)
- imbalanced-learn (0.7.0)
- matplotlib (3.3.1)
- seaborn (0.11.0)
- pydot_ng (2.0.0)

##### Python scripts
-----------------------
##### Main scripts (in chronological order as used in the study)
- `subtype_hmm.py` : Use hidden Markov models (HMM) to discriminate GH13 functional subtypes (CBH vs EG)
- `subtype_rules.py`: Derive position-specific classification rules for discriminating GH13 functional subtypes

##### Other Python scripts
- `bioinformatics.py`: contains adhoc functions for bioinformatic analysis
- `plots_and_analysis.py`: for analyzing results and plotting the figures in the manuscript 

## Other softwares (version used in this work)
- HMMER v3.1b2
- MAFFT v7.475 (2020/Nov/23)
- NCBI command-line BLAST v2.11.0



## Datasets and plots
-------------------------
- Sequence datasets are in `fasta/`
- Sequences split into five folds used for validation and design of the HMM, as well as the final trained HMMs, are in `hmm_train_test/` 
- Datasets containing results presented in the paper are in `results_final/`
- Figures and tables in the manuscript are in `plots/`
