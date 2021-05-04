"""
Discriminating GH13 ASs and SHs with Random Forest.
"""




# Imports
#=====================#
import pandas as pd
import numpy as np
from scipy import stats
import random
from Bio import SeqIO
import os
import subprocess

from imblearn.under_sampling import RandomUnderSampler

from Bio.Blast.Applications import NcbiblastpCommandline

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

import bioinformatics as bioinf



# Prepare sequences and data
#=====================================================#
GH13_df = pd.read_csv('results_final/ncbi_subtypes.csv')
GH13_SH = GH13_df[(GH13_df.ncbi_pred_class==0)]
accession_SH = GH13_SH.Accession.tolist() 
accession_all = bioinf.get_accession('fasta/initial_blast/nrblast_all.fasta')
GH13 = [1 if x in accession_SH else 0 for x in accession_all]
# class labels
y = pd.Series(GH13)   
GH13_not_SH = y[y==0]  
GH13_yes_SH = y[y==1]

# Derive features for machine learning with one-hot encoding
#============================================================#
cat_domain_fasta = 'fasta/GH13_positions_only/GH13_cat.fasta'
sequence_df = bioinf.fasta_to_df(cat_domain_fasta)
X_features = pd.DataFrame() # empty dataframe for storing features

for i in range(len(sequence_df.columns)):
    # Convert amino acids to integers
    X_resid = list(sequence_df.iloc[:,i])
    labelencoder = LabelEncoder()
    X_label = list(labelencoder.fit_transform(X_resid))
    X_resid_unique = sorted(set(X_resid))
    X_label_unique = sorted(set(X_label))
    
    # Map integer labels to amino acids
    label_resid = [X_label.index(num) for num in X_label_unique]
    label_resid = [X_resid[num] for num in label_resid]
    
    # Convert labels to binary features (one-hot encoding)
    onehotencoder = OneHotEncoder()
    X_label = pd.DataFrame(X_label) # convert to 2D array
    X_encoded = onehotencoder.fit_transform(X_label).toarray()
    X_encoded = pd.DataFrame(X_encoded)
    
    # Name encoded features (residue + position, e.g G434)
    X_encoded.columns = ['{0}{1}'.format(res,i+1) for res in label_resid]
    del X_encoded['-{0}'.format(i+1)]  # remove encoded features from gaps
    
    # Append features to dataframe store
    for col in X_encoded.columns:
        X_features[col] = X_encoded[col]    

		
# Randomly split data to validation set and test set
#====================================================#


# Test set data (10% of total data)
SH_test_size = int(0.1 * len(GH13_yes_SH))
AS_test_size = int(0.1 * len(GH13_not_SH))
SH_test_indices = random.sample(list(GH13_yes_SH.index), SH_test_size)
AS_test_indices = random.sample(list(GH13_not_SH.index), AS_test_size)
test_indices = SH_test_indices + AS_test_indices
test_indices = sorted(test_indices)

# Validation set data (90% of total data)
val_indices = [x for x in list(y.index) if x not in test_indices]

# X (features) and y for validation and test sets
X_val = X_features.iloc[val_indices,:]
y_val = y.iloc[val_indices]
X_test_sep = X_features.iloc[test_indices,:]
y_test_sep = y.iloc[test_indices]






# Apply random forests to validation set using all features
#=============================================================#

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# Function for evaluating performance
def evalPerf(y_test, y_pred):
    '''Return (sensitivity, specificity, accuracy, MCC, p_value)'''
    cm = confusion_matrix(y_test, y_pred)
    tn, tp, fn, fp = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
    n = tp + fp + tn + fn
    accuracy = (tp + tn)/n * 100
    mcc = ((tp*tn) - (fp*fn))/np.sqrt((tp+fp)*(tn+fn)*(tp+fp)*(tn+fp))
    sens = tp/(tp + fn) * 100 if tp + fp != 0 else 0
    spec = tn/(tn + fp) * 100 if tn + fn != 0 else 0
    table = np.array([[tp, fp], [fn, tn]]) # CBH and EG have same contingency table
    p_value = stats.chi2_contingency(table)[1]
    return [sens, spec, accuracy, mcc, p_value]


# 100 repetitions of 10-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=800, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store = pd.DataFrame(store, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_mean = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std = pd.DataFrame(featimp_store).std(axis=0)
store_featimp = pd.DataFrame([X_val.columns, featimp_mean, featimp_std], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to spreadsheet
store.to_csv('results_final/ml_rf_pred/perf_all.csv')
store_featimp.to_csv('results_final/ml_rf_pred/featimp_all.csv')
     
   
        


        
# Use only top 50 features
#===================================#

# Top 50 features
top50_index = list(store_featimp.sort_values(by='mean', ascending=False).iloc[:50,:].index)
X_val_top50 = X_val.iloc[:,top50_index]

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# 100 repetitions of 10-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val_top50, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=800, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store_top50 = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store_top50 = pd.DataFrame(store_top50, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_mean_top50 = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std_top50 = pd.DataFrame(featimp_store).std(axis=0)
store_featimp_top50 = pd.DataFrame([X_val_top50.columns, featimp_mean_top50, featimp_std_top50], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to spreadsheet
store_top50.to_csv('results_final/ml_rf_pred/perf_top50.csv')
store_featimp_top50.to_csv('results_final/ml_rf_pred/featimp_top50.csv')


# Use only top 20 features
#=================================#

# Top 20 features
top20_index = list(store_featimp.sort_values(by='mean', ascending=False).iloc[:20,:].index)
X_val_top20 = X_val.iloc[:,top20_index]

# Empty lists for storing final results
sens_store, spec_store, acc_store, mcc_store, featimp_store = [], [], [], [], []

# 100 repetitions of 10-fold cross validation
for r in range(100):
    RUS = RandomUnderSampler(random_state=None)
    X_select, y_select = RUS.fit_resample(X_val_top20, y_val)
    X_select, y_select = pd.DataFrame(X_select), pd.Series(y_select)
    
    # 5-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=None)
    kf_indices = kf.split(X_select)
    for train_index, test_index in kf_indices:
        X_train, y_train = X_select.iloc[train_index, :], y_select.iloc[train_index]
        X_test, y_test = X_select.iloc[test_index, :], y_select.iloc[test_index]
        
        # Fit random forest classifier to training data
        classifier = RandomForestClassifier(n_estimators=800, n_jobs=-1)
        classifier.fit(X_train, y_train)
        
        # Test classifier and evaluate performance
        y_pred = classifier.predict(X_test)
        sens, spec, accuracy, mcc, pvalue = evalPerf(y_test, y_pred)
        featimp = list(classifier.feature_importances_)
        
        # Save results
        sens_store.append(sens)
        spec_store.append(spec)
        acc_store.append(accuracy)
        mcc_store.append(mcc)
        featimp_store.append(featimp)

# Average results over all 500 repetitions
store_top20 = [np.mean(sens_store), np.std(sens_store), np.mean(spec_store), np.std(spec_store),
         np.mean(acc_store), np.std(acc_store), np.mean(mcc_store), np.std(mcc_store)]
store_top20 = pd.DataFrame(store_top20, index=['sens_mean', 'sens_std', 'spec_mean', 'spec_std',
                                   'acc_mean', 'acc_std', 'mcc_mean', 'mcc_std'])
featimp_data = pd.DataFrame(featimp_store, columns=X_val_top20.columns)
featimp_mean_top20 = pd.DataFrame(featimp_store).mean(axis=0)
featimp_std_top20 = pd.DataFrame(featimp_store).std(axis=0)
store_featimp_top20 = pd.DataFrame([X_val_top20.columns, featimp_mean_top20, featimp_std_top20], 
                             index=['features', 'mean', 'std']).transpose()

# Write results to excel spreadsheet
featimp_data.to_csv('results_final/ml_rf_pred/featimp_top20_fulldata.csv')
store_top20.to_csv('results_final/ml_rf_pred/perf_top20.csv')
store_featimp_top20.to_csv('results_final/ml_rf_pred/featimp_top20.csv')
        



# Train top20 classifier and test on test set
#===============================================#
RUS = RandomUnderSampler(random_state=None)
X_select, y_select = RUS.fit_resample(X_val_top20, y_val)
classifier = RandomForestClassifier(n_estimators=800, n_jobs=-1)
classifier.fit(X_select, y_select)
y_pred = classifier.predict(X_test_sep.iloc[:,top20_index])
cm = confusion_matrix(y_test_sep, y_pred)
tn, tp, fn, fp = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
sens, spec, accuracy, mcc, pvalue = evalPerf(y_test_sep, y_pred)
store = pd.DataFrame([tp, fp, tn, fn, sens, spec, accuracy, mcc], 
                     index=['tp', 'fp', 'tn', 'fn', 'sens', 'spec', 'acc', 'mcc'])
store.to_csv('results_final/ml_rf_pred/perf_test_set.csv')

        




# Position specific rules from top 50 features (Supporting Information)
#===============================================================================#
store = [] # empty list for storing results
for col in X_val_top50.columns:
    # Test the rule, feature => SH
    y_pred = X_val_top50[col]
    cm = confusion_matrix(y_val, y_pred)
    perf = evalPerf(y_val, y_pred)
    
    # Test the rule, not feature => SH
    y_pred1 = [1 if x==0 else 0 for x in y_pred]
    cm1 = confusion_matrix(y_val, y_pred1)
    perf1 = evalPerf(y_val, y_pred1)
    
    # Add rule (i.e. the rule with the highest MCC)
    if perf[3] > perf1[3]:
        perf = ['{0}=>SH'.format(col)] + perf
        store.append(perf)
    else:
        perf1 = ['not {0}=>SH'.format(col)] + perf1
        store.append(perf1)

store = pd.DataFrame(store, columns=['rule', 'sensitivity', 
                                     'specificity', 'accuracy', 'mcc', 'pvalue'])
store.to_csv('results_final/ml_rf_pred/position_rules.csv')

    
    
    






