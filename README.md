# GH13_ML
## Deciphering the mechanism of transglycosylation and hydrolysis in a GH13 family using machine learning 
----------------

## Python version 
-----------------
- Python (3.6.8)

#### Python modules (version used in this work)
- pandas (1.1.5)
- numpy (1.19.1)
- scipy (1.5.4)
- biopython (1.78)
- scikit-learn (0.24.1)
- imbalanced-learn (0.7.0)
- matplotlib (3.3.4)
- seaborn (0.11.1)
- pydot_ng (2.0.0)

#### Python scripts
-----------------------
#### Main scripts
- `subtype_hmm.py` : Use hidden Markov models (HMM) to discriminate GH13 functional subtypes (ASs vs SHs)
- `subtype_rules.py`: Derive position-specific classification rules for discriminating GH13 functional subtypes
- `subtype_rf.py`: Using random forest classifier to distinguish the GH13 functional subtypes
- 
#### Other Python scripts
- `bioinformatics.py`: contains adhoc functions for bioinformatic analysis
- `plots_and_analysis.py`: for analyzing results and plotting the figures

## Other softwares
- HMMER v3.1b2
- MAFFT v7.475 (2020/Nov/23)
- NCBI command-line BLAST v2.11.0



## Datasets and plots
-------------------------
- Sequence datasets are in `fasta/`
- Sequences split into five folds used for validation and design of the HMM, as well as the final trained HMMs, are in `hmm_train_test/` 
- Datasets containing results presented in the paper are in `results_final/`
- Figures and tables are in `plots/`
-------------------------
## Reference

Gado, J.E., Harrison, B.E., Sandgren, M., Ståhlberg, J., Beckham, G.T., and Payne, C.M. **Machine learning reveals sequence-function relationships in family 7 glycoside hydrolases.** Submitted to *FEBS* (2020).
