"""
Prediction of GH13 subtypes (AS/SH) with hidden Markov models (HMM)
"""




# Imports
#=============#

import pandas as pd
import numpy as np
import subprocess

from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

import bioinformatics as bioinf






# Split sequences into 10 folds for training/testing HMM
#============================================================#

def kfold_split(fasta, k, path, root='kfold'):
    '''Spilt sequences in fasta file into k-folds and save them as k-separate fasta files
    for k-fold training and testing (saved as ${root}_train1.fasta, ${root}_test1.fasta, 
    etc.)'''
    
    [h,s] = bioinf.split_fasta(fasta)
    kf = KFold(n_splits=10, random_state=0, shuffle=True)
    dummy=0
    for train_index, test_index in kf.split(range(len(s))):
        dummy += 1
        bioinf.combine_fasta([h[x] for x in train_index], [s[x] for x in train_index], 
                             f'{path}/{root}_train{dummy}.fasta')
        bioinf.combine_fasta([h[x] for x in test_index], [s[x] for x in test_index], 
                             f'{path}/{root}_test{dummy}.fasta')

    
# NCBI sequences (600 total, 300 AS, 300 SH)
AS_file = 'fasta/subtype/alignments/ASs_msa_s.fasta'
SH_file = 'fasta/subtype/alignments/SHs_msa_s.fasta'

kfold_split(AS_file, k=10, path='hmm_train_test/ncbi_kfold', root='AS')
kfold_split(SH_file, k=10, path='hmm_train_test/ncbi_kfold', root='SH')






# Train and test HMM on 10 folds of sequences
#=================================================#t

def implement_hmm_bash(path, k=10):
    '''Implement HMM with Python's subprocess function. HMMMER should be installed on your
    machine and the commands, hmmbuild and hmmsearch, should be callable from the command
    line.  Train/test fasta sequences should be present in $path (as AS_train1.fasta, 
    AS_test1.fasta, SH_train1.fasta, SH_test1.fasta, etc.) HMMs are tested on a 
    left-out fold, and trained on all other folds.'''
    
    # Loop through all k folds, train HMMs on train fold sequences
    # and test on test fold sequences
    for i in range(1, k+1):
        # Build/train HMM
        AS_train_hmm = f'{path}/AS{i}.hmm'
        AS_train_fasta = f'{path}/AS_train{i}.fasta'
        SH_train_hmm = f'{path}/SH{i}.hmm'
        SH_train_fasta = f'{path}/SH_train{i}.fasta'
        AS_command = f'hmmbuild {AS_train_hmm} {AS_train_fasta}'
        SH_command = f'hmmbuild {SH_train_hmm} {SH_train_fasta}'
        
        subprocess.call(AS_command, shell=True)
        subprocess.call(SH_command, shell=True)

        # Test HMM on AS and SH test fold
        AS_test_fasta = f'{path}/AS_test{i}.fasta'
        SH_test_fasta = f'{path}/SH_test{i}.fasta'
        AS_to_AS_out = f'{path}/AShmm_AS{i}.out'
        AS_to_SH_out = f'{path}/AShmm_SH{i}.out'
        SH_to_AS_out = f'{path}/SHhmm_AS{i}.out'
        SH_to_SH_out = f'{path}/SHhmm_SH{i}.out'
        AS_to_AS_command = f'hmmsearch -T 0 --incT 0 --nobias {AS_train_hmm} ' \
                             f'{AS_test_fasta} > {AS_to_AS_out}'
        AS_to_SH_command = f'hmmsearch -T 0 --incT 0 --nobias {AS_train_hmm} ' \
                             f'{SH_test_fasta} > {AS_to_SH_out}'
        SH_to_AS_command = f'hmmsearch -T 0 --incT 0 --nobias {SH_train_hmm} ' \
                             f'{AS_test_fasta} > {SH_to_AS_out}'
        SH_to_SH_command = f'hmmsearch -T 0 --incT 0 --nobias {SH_train_hmm} ' \
                             f'{SH_test_fasta} > {SH_to_SH_out}'
        
        subprocess.call(AS_to_AS_command, shell=True)
        subprocess.call(AS_to_SH_command, shell=True)
        subprocess.call(SH_to_AS_command, shell=True)
        subprocess.call(SH_to_SH_command, shell=True)



    
# NCBI sequences (600 total, 300 AS, 300 SH)
implement_hmm_bash(path='hmm_train_test/ncbi_kfold', k=10)






# Retrieve and compare HMM scores from HMM output files, store results
#=======================================================================#

def get_acc_and_scores(hmm_output):
    '''Read an HMMER align output file and retrieve
    the accession codes and HMM scores in the file.
    Return the list, [accessions, scores].'''
    
    with open(hmm_output,'r') as file:
        text = file.read()
    text = text[text.index('E-value'):text.index('Domain')]
    text_lines = text.split('\n')[2:-3]
    text_lines = [x for x in text_lines if '---' not in x]
    scores = [line.split()[1] for line in text_lines]
    scores = [float(x) for x in scores]
    accessions = [line.split()[8] for line in text_lines]
    return [accessions, scores]



def compare_hmmscores(fasta, hmm1, hmm2):
    '''Return a dataframe whose columns are the accession 
    numbers in fasta and the corresponding hmm scores from 
    the hmm output files, hmm1 and hmm2, respectively.'''
    
    acc_all = bioinf.get_accession(fasta)
    [acc1, score1] = get_acc_and_scores(hmm1)
    [acc2, score2] = get_acc_and_scores(hmm2)
    hmm1_scores,hmm2_scores = [],[]
    for i in range(len(acc_all)):
        try:
            hmm1_scores.append(score1[acc1.index(acc_all[i])])
        except:
            hmm1_scores.append(0)  # Assign a score of 0 if it's below the threshold
            
        try:
            hmm2_scores.append(score2[acc2.index(acc_all[i])])
        except:
            hmm2_scores.append(0)
    store = pd.DataFrame([acc_all, hmm1_scores, hmm2_scores]).transpose()
    store.columns = ['accession', 'hmm1_scores', 'hmm2_scores']
    return store

# NCBI sequences (600 total, 300 AS, 300 SH)
AS_store = pd.DataFrame()
SH_store = pd.DataFrame()
path = 'hmm_train_test/ncbi_kfold/'
k = 10
for i in range(k):
    fasta_AS = path + f'AS_test{i+1}.fasta'
    hmm1_AS = path + f'AShmm_AS{i+1}.out'
    hmm2_AS = path + f'SHhmm_AS{i+1}.out'
    AS_store = AS_store.append(compare_hmmscores(fasta_AS, hmm1_AS, hmm2_AS), 
                                 ignore_index=True)
    
    fasta_SH = path + f'SH_test{i+1}.fasta'
    hmm1_SH = path + f'AShmm_SH{i+1}.out'
    hmm2_SH = path + f'SHhmm_SH{i+1}.out'
    SH_store = SH_store.append(compare_hmmscores(fasta_SH, hmm1_SH, hmm2_SH), 
                                 ignore_index=True)

AS_store['diff_score'] = pd.Series(np.array(AS_store.iloc[:,1]) - 
                                     np.array(AS_store.iloc[:,2]))
AS_store['true_class'] = pd.Series([1]*len(AS_store))
AS_store['pred_class'] = pd.Series([1 if x > 0 else 0 for x in AS_store.iloc[:,3]])
SH_store['diff_score'] = pd.Series(np.array(SH_store.iloc[:,1]) - 
                                     np.array(SH_store.iloc[:,2]))
SH_store['true_class'] = pd.Series([0]*len(SH_store))
SH_store['pred_class'] = pd.Series([1 if x > 0 else 0 for x in SH_store.iloc[:,3]])
store = AS_store.append(SH_store, ignore_index=True)
store.index=store.index+1
store.to_csv('results_final/ncbi_kfold.csv')






# Train final HMMs on all data (all 10 folds), store for future use
#===================================================================#

# NCBI HMMs
ncbi_AS_fasta = 'fasta/subtype/alignments/ASs_msa_s.fasta'
ncbi_SH_fasta = 'fasta/subtype/alignments/SHs_msa_s.fasta'

ncbi_AS_hmm = 'hmm_train_test/final_hmm/AS_ncbi.hmm'
ncbi_SH_hmm = 'hmm_train_test/final_hmm/SH_ncbi.hmm'
AS_command = f'hmmbuild {ncbi_AS_hmm} {ncbi_AS_fasta}'
SH_command = f'hmmbuild {ncbi_SH_hmm} {ncbi_SH_fasta}'
subprocess.call(AS_command, shell=True)
subprocess.call(SH_command, shell=True)
    





# Apply NCBI HMM and UniProtKB/Swiss-Prot HMM to 4,991 sequences
#================================================================#

# File paths
fastafile = 'fasta/initial_blast/nrblast_all.fasta'

ncbi_AS_out = 'hmm_train_test/final_hmm/hmm_to_4991/all_AS_ncbi.out'
ncbi_SH_out = 'hmm_train_test/final_hmm/hmm_to_4991/all_SH_ncbi.out'

# Commands

ncbi_AS_command = f'hmmsearch -T 0 --incT 0 --nobias {ncbi_AS_hmm} {fastafile} > ' \
                   f'{ncbi_AS_out}'
ncbi_SH_command = f'hmmsearch -T 0 --incT 0 --nobias {ncbi_SH_hmm} {fastafile} > ' \
                   f'{ncbi_SH_out}'

# Run commands to align 4,991 sequences to HMM

subprocess.call(ncbi_AS_command, shell=True)
subprocess.call(ncbi_SH_command, shell=True)


# Collect results and write to spreadsheet

ncbi_results = pd.DataFrame()

ncbi_results = ncbi_results.append(compare_hmmscores(fastafile, ncbi_AS_out, ncbi_SH_out), 
                                 ignore_index=True)
ncbi_results['diff_score'] = pd.Series(np.array(ncbi_results.iloc[:,1]) - 
                                      np.array(ncbi_results.iloc[:,2]))
ncbi_results['pred_class'] = pd.Series([1 if x > 0 else 0 for x in ncbi_results.iloc[:,3]])

ncbi_results.columns = ['Accession', 'ncbi_AS_scores', 'ncbi_SH_scores', 
                       'ncbi_diff_scores', 'ncbi_pred_class']

ncbi_results.index=ncbi_results.index+1
ncbi_results.to_csv('results_final/ncbi_subtypes.csv')

