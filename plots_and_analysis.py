"""
Analyze results and plot figures
"""




# Imports
#==============#

import pandas as pd
import numpy as np
import scipy
import random

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import bioinformatics as bioinf
 



# Plots for HMM method 10-fold cross validation 
#===============================================#

fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'18'}
legend_font = {'family':fnt, 'size':'12'}
label_font = {'family':fnt, 'size':'20'}
plt.rcParams['figure.figsize'] = [6,3]

ec = 'black'
legend_label = ['AS', 'SH']



# NCBI dataset
ex = pd.read_csv('results_final/ncbi_kfold.csv')
lw = 0.10
ASs = list(ex.diff_score[:300])
SHs = list(ex.diff_score[300:])
random.shuffle(ASs)
random.shuffle(SHs)
out1 = plt.bar(range(300), ASs, color='blue', linewidth=lw, 
               edgecolor='blue')
out2 = plt.bar(range(300,600), SHs, color='red', linewidth=lw, 
               edgecolor='red')
pltout = [x[0] for x in [out1, out2]]

plt.xlabel('Sequence', **label_font)
plt.ylabel('Score difference', **label_font)
plt.xticks(**ticks_font)
plt.yticks([-300,-150,0,150,300], **ticks_font)
plt.xlim([-1,601])
plt.axhline(color='black', linewidth=1)
plt.legend(pltout, legend_label, prop=legend_font, 
           loc='upper right')
plt.tight_layout()
plt.savefig('plots/ncbi_kfold.pdf')
plt.close()


# Table of classification/association rules
#===========================================#

from subtype_rules import GH13MSA

ASmsa = 'fasta/GH13_positions_only/AS_cat.fasta'
SHmsa = 'fasta/GH13_positions_only/SH_cat.fasta'
GH13msa = GH13MSA(ASmsa, SHmsa)
GH13msa.get_freq(include_gaps=True)
rules = pd.read_csv('results_final/rules/rules_all.csv', index_col=0)
rules_amino = pd.read_csv('results_final/rules/rules_amino.csv', index_col=0)
rules_type = pd.read_csv('results_final/rules/rules_type.csv', index_col=0)

mcc = list(rules.mcc)
min_mcc = np.percentile(mcc, 95)  # mcc > 0.73
rules_mcc = rules[rules.mcc >= min_mcc]
rules_amino_mcc = rules_amino[rules_amino.mcc >= min_mcc]  # 72 rules
rules_type_mcc = rules_type[rules_type.mcc >= min_mcc]  # 33 rules
positions = sorted(set(rules_mcc.tre_pos))  # 61 positions
rules_mcc.to_csv('results_final/rules/rules_mcc.csv')
rules_amino_mcc.to_csv('results_final/rules/rules_amino_mcc.csv')
rules_type_mcc.to_csv('results_final/rules/rules_type_mcc.csv')

rules_amino_table = rules_amino_mcc.loc[:,['tre_pos','rule', 'closest_subsite', 
                                           'dist_subsite','sens', 'spec', 'acc', 'mcc']]
rules_amino_table.columns = ['Position', 'Rule', 'Closest subsite', 
                             'Distance to closest subsite (Å)', 'Sensitivity', 
                             'Specificity', 'Accuracy', 'MCC']
rules_amino_table.to_csv('plots/rules_amino_table.csv')
rules_type_table = rules_type_mcc.loc[:,['tre_pos','rule', 'closest_subsite', 
                                         'dist_subsite', 'sens', 'spec', 'acc', 'mcc']]
rules_type_table.columns = ['Position', 'Rule', 'Closest subsite', 
                             'Distance to closest subsite (Å)', 'Sensitivity',
                             'Specificity', 'Accuracy', 'MCC']
rules_type_table.to_csv('plots/rules_type_table.csv')






# Plot Histogram for  MCC of rules
#=================================#

fnt='Arial'
ticks_font = {'fontname':fnt, 'size':'20'}
label_font = {'family':fnt, 'size':'22'}
title_font = {'family':fnt, 'size':'24'}
plt.rcParams['figure.figsize'] = [6,3.5]
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['axes.axisbelow'] = True

weights = np.zeros_like(mcc) + 1/len(mcc)
plt.hist(mcc, bins=12, rwidth=1, color='darkgreen', weights=weights)
plt.xticks(np.arange(-80,101,40)*0.01, **ticks_font)
plt.yticks(np.arange(0,28,5)*0.01, **ticks_font)
plt.xlabel('MCC', **label_font)
plt.ylabel('Relative frequency', **label_font)
plt.tight_layout()
plt.savefig('plots/rules_mcc_dist.pdf')






# Minimum distance between rules' positions and substrate
#============================================================#

dist50 = np.percentile(rules_mcc.dist_subsite, 50) #4.79Å
rule_dist = list(rules_mcc.dist_subsite)
weights = np.zeros_like(rule_dist) + 1/len(rule_dist)
plt.hist(rule_dist, bins=7, weights=weights, color='indigo')
plt.xticks(np.arange(0,30,5), **ticks_font)
plt.xlim((0,25))
plt.yticks(**ticks_font)
plt.xlabel('Distance to substrate (Å)', **label_font)
plt.ylabel('Relative frequency', **label_font)
plt.tight_layout()
plt.savefig('plots/rules_distance_dist.pdf')









# Distribution at 61 important positions
#==========================================#

plt.rcParams['figure.figsize'] = [7,4]
for i in range(len(positions)):
    GH13msa.site_plot(site=positions[i], savefig=True, 
                      savepath='plots/position_distribution')






# Aromatic residues within 6Å of substrate (and consensus AS and SH)
#==============================================================================#

GH13msa.get_consensus_sequences()
AS_consensus = list(GH13msa.consensus_AS)
SH_consensus = list(GH13msa.consensus_SH)
tre = bioinf.split_fasta('fasta/GH13_positions_only/consensus.fasta')[1][1]
excel = pd.read_csv('results_final/residue_distances.csv', index_col=0)
closest_subsite = list(excel.iloc[:,0])
distances = list(excel.iloc[:,1])

resid_aro, tre_aro, AS_aro, SH_aro, closest_subsite_aro, dist_aro = [],[],[],[],[],[]
AS_aro_freq, SH_aro_freq, conserved = [], [], []
aro_res = ['F', 'W', 'Y', 'H']

for i in range(len(tre)):
    if (tre[i] in aro_res or AS_consensus[i] in  aro_res or SH_consensus[i] in aro_res)\
    and distances[i]<=6.0:
        resid_aro.append(i+1)
        tre_aro.append(tre[i])
        AS_aro.append(AS_consensus[i])
        SH_aro.append(SH_consensus[i])
        closest_subsite_aro.append(closest_subsite[i])
        dist_aro.append(distances[i])
        AS_freq = GH13msa.AS_freq.iloc[[4,6,18,19],i].sum()*100
        SH_freq = GH13msa.SH_freq.iloc[[4,6,18,19],i].sum()*100
        AS_aro_freq.append(AS_freq)
        SH_aro_freq.append(SH_freq)
        if AS_freq > 66 and SH_freq < 66:
            conserved.append('AS')
        elif AS_freq < 66 and SH_freq > 66:
            conserved.append('SH')
        elif AS_freq > 49 and SH_freq > 49:
            conserved.append('AS and SH')
        else:
            conserved.append('None')

store = pd.DataFrame([resid_aro, tre_aro, AS_aro, SH_aro, AS_aro_freq, SH_aro_freq, 
                      closest_subsite_aro, dist_aro, conserved]).transpose()
store.columns = ['Position', 'GH13 residue', 'AS consensus residue', 
                 'SH consensus residue', 'Frequency of aromatic residues in ASs (%)', 
                 'Frequency of aromatic residues in SHs (%)', 'Closest subsite', 
                 'Distance to closest subsite (Å)', 'Aromatic residues conserved (>66%) in']
store = store.sort_values('Closest subsite')
store.to_csv('results_final/aromatic_residues.csv')






# Pymol commands for viewing aromatic residues on structure
#=============================================================#
pymol_AS = 'select aroAS, '
pymol_both = 'select aroboth, '
for i in range(len(store)):
    pos = store.iloc[i,0]
    if store.iloc[i,-1]=='AS':
        pymol_AS += f'resi {pos} or '
    elif store.iloc[i,-1]=='AS and SH':
        pymol_both += f'resi {pos} or '
with open('plots/aromatic_pymol.txt', 'w') as pym:
    pym.write(pymol_AS[:-4] + '\n\n')
    pym.write(pymol_both[:-4] + '\n\n')

