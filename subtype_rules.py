"""
Discriminating GH13 ASs and SHs with position-specific classification rules
"""




# Imports
#================#
import pandas as pd
import numpy as np
from scipy import stats

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Selection
import matplotlib.pyplot as plt

import bioinformatics as bioinf

import warnings
warnings.filterwarnings("ignore")






# Prepare sequences and data
#=================================#
if __name__ == '__main__':
    # Get MSA with only GH13 positions for analysis
    heads, sequences = bioinf.split_fasta('fasta/subtype/alignments/nrblast_all_msa_s.fasta')
    GH13_seq = sequences[0]
    GH13_positions = [x for x in range(len(GH13_seq)) 
                            if GH13_seq[x].isalpha()]
    sequences_treonly = []
    for i in range(len(sequences)):
        seq = list(sequences[i])
        seq = [seq[x] for x in GH13_positions]
        seq = ''.join(seq)
        sequences_treonly.append(seq)
    bioinf.combine_fasta(heads, sequences_treonly, 'fasta/GH13_positions_only/' \
                         'GH13_all.fasta')
    
    
    # Separate sequences in MSA to two sub-MSAs (AS and SH)
    subtype = list(pd.read_csv('results_final/ncbi_subtypes.csv')['ncbi_pred_class'])
    AS_pos = [x for x in range(len(subtype)) if subtype[x]==1]
    SH_pos = [x for x in range(len(subtype)) if subtype[x]==0]
    heads_AS = [heads[x] for x in AS_pos]
    
    sequences_AS = [sequences_treonly[x] for x in AS_pos]
    bioinf.combine_fasta(heads_AS, sequences_AS, 'fasta/GH13_positions_only/' \
                         'AS_all.fasta')
    
    heads_SH = [heads[x] for x in SH_pos]
    sequences_SH = [sequences_treonly[x] for x in SH_pos]
    bioinf.combine_fasta(heads_SH, sequences_SH, 'fasta/GH13_positions_only/' \
                         'SH_all.fasta')
    
    # Save MSA of only catalytic domain
    sequences_cat = [seq[98:551] for seq in sequences_treonly]
    bioinf.combine_fasta(heads, sequences_cat, 'fasta/GH13_positions_only/' \
                         'GH13_cat.fasta')
    
    seq_cat_AS = [sequences_cat[x] for x in AS_pos]
    bioinf.combine_fasta(heads_AS, seq_cat_AS, 'fasta/GH13_positions_only/' \
                         'AS_cat.fasta')
    
    seq_cat_SH = [sequences_cat[x] for x in SH_pos]
    bioinf.combine_fasta(heads_SH, seq_cat_SH, 'fasta/GH13_positions_only/' \
                         'SH_cat.fasta')






# Create a Class for efficient analysis of MSA
#================================================#

class GH13MSA():
    '''A class for efficient analyses of GH13 MSA, and for 
        deriving position-specific classification rules. 
        AS_msa is the fasta file of the subalignment containing 
        AS sequences and only GH13 positions. SH_msa if the
        fasta file for SH sequences.'''
        
        
    def __init__(self, AS_msa, SH_msa):
        self.AS_msa = AS_msa
        self.SH_msa = SH_msa
        self._AS_color = 'blue'
        self._SH_color = 'red'
        self.AS_size = len(bioinf.split_fasta(self.AS_msa)[1])
        self.SH_size = len(bioinf.split_fasta(self.SH_msa)[1])
        
        
    def _get_aa_freq(self, fasta, analysis='amino', include_gaps=True):
        '''Return a dataframe of the frequencies of all 20 amino acids (AAs)
        in each site of an MSA. If include_gaps=True, gaps are treated as
        AAs and are included in the analysis. 
        If analysis == 'amino', frequencies of AAs are computed.
        if analysis == 'type', frequencies of AA types are computed'''
        
        if analysis=='amino':
            fasta_df = bioinf.fasta_to_df(fasta)
            amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        elif analysis=='type':
            # Replace AA single letter with single letter describing
            # the AA type
            # Aliphatic (A), Aromatic (R), Polar (P), Positve (T), 
            # and Negative (N)
            fasta_df = bioinf.residue_to_group(fasta)
            amino_acids = list('ARPTN') 
        if include_gaps:
            amino_acids += ['-']
            
        # Determine frequency
        store = []
        length = len(fasta_df.index)
        for k in range(len(fasta_df.columns)):
            aa_list = list(fasta_df.iloc[:,k])
            aa_count = [aa_list.count(x)/length for x in amino_acids]
            store.append(aa_count)
        store = pd.DataFrame(store).transpose()
        store.index = amino_acids
        return store
        
    
    def get_freq(self, analysis='amino', include_gaps=True):
        '''Determine the amino acid frequencies of positions
        in AS and SH subalignments.'''
        
        self.AS_freq = self._get_aa_freq(self.AS_msa, analysis='amino', 
                                          include_gaps=include_gaps)
        self.SH_freq = self._get_aa_freq(self.SH_msa, analysis='amino', 
                                          include_gaps=include_gaps)

            
    def get_consensus_sequences(self):
        '''Determine the consensus sequence for AS and SH from
        the MSAs.'''
        
        AS_cons, SH_cons = '', '' # Initialize empty string
        amino_acids = list(self.AS_freq.index)
        
        # Loop through each position and determine the most frequent amino acid
        for i in range(len(self.AS_freq.columns)):
            c_freq = list(self.AS_freq.iloc[:,i])
            e_freq = list(self.SH_freq.iloc[:,i])
            AS_cons += amino_acids[c_freq.index(max(c_freq))]
            SH_cons += amino_acids[e_freq.index(max(e_freq))]
        self.consensus_AS = AS_cons
        self.consensus_SH = SH_cons
    
    def _one_to_three(self, one):
        '''Convert one-letter amino acid to three'''
        ones = list('ACDEFGHIKLMNPQRSTVWY')
        threes = ['Ala', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His', 'Ile', 'Lys', 'Leu',
                  'Met', 'Asn', 'Pro', 'Gln', 'Arg', 'Ser', 'Thr', 'Val', 'Trp', 'Tyr']
        return threes[ones.index(one)]
    
    
    def get_rules(self, analysis='amino'):
        '''Derive position-specific classification rules for discriminating 
        GH13 ASs from SHs using the consensus residue (or residue type) 
        from the MSA.'''
        
        [AS_freq, SH_freq] = [self._get_aa_freq(x, analysis=analysis, include_gaps=True) 
                                for x in [self.AS_msa, self.SH_msa]]
        if analysis=='type':
            ind = ['ALI', 'ARO', 'POL', 'POS', 'NEG', '-']
            AS_freq.index, SH_freq.index = ind, ind
            
        # Empty lists for storing results
        pos, sens, spec, acc, mcc, rule, pvalue = [], [], [], [], [], [], []
        
        # Loop through each position, derive rules, and test the rules
        for i in range(len(AS_freq.columns)):
            AS_cons_freq = AS_freq.iloc[:,i].max() # Frequency of consensus AA/type
            AS_cons_type = AS_freq.index[list(AS_freq.iloc[:,i]).index(AS_cons_freq)]
            SH_cons_freq = SH_freq.iloc[:,i].max()
            SH_cons_type = SH_freq.index[list(SH_freq.iloc[:,i]).index(SH_cons_freq)]
            
            # Rule 1: [X ==> AS, not X ==> SH]
            if AS_cons_type != '-':
                sensitivity = AS_cons_freq   # X => AS
                cons_pos = list(AS_freq.index).index(AS_cons_type)
                specificity = 1 - SH_freq.iloc[cons_pos,i] # not X => SH
                accuracy = (sensitivity*self.AS_size + specificity*self.SH_size)/(self.AS_size + self.SH_size)
                tp = sensitivity * self.AS_size  # X => AS
                fp = self.SH_size*SH_freq.iloc[cons_pos,i] # X => SH
                tn = specificity*self.SH_size  # not X => SH
                fn = self.AS_size * (1 - AS_freq.iloc[cons_pos,i]) # not X => AS
                MCC = ((tp * tn) - (fp * fn))/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
                if tp == 1 or fp == 0 or fn ==0 or tn == 1:
                    p_value = 0
                else:					
                    table = np.array([[tp, fp], [fn, tn]]) # AS and SH have same contingency table
                    p_value = stats.chi2_contingency(table)[1]
                
                pos.append(i+1) # Position in GH13
                sens.append(sensitivity * 100)
                spec.append(specificity * 100)
                acc.append(accuracy * 100)
                mcc.append(MCC)
                pvalue.append(p_value)
                key = self._one_to_three(AS_cons_type) if analysis=='amino' else AS_cons_type
                rule.append(f'{key}=>AS, not {key}=>SH')
            
            
            # Rule 2: [Z ==> SH, not Z ==> AS]
            if SH_cons_type != '-':
                cons_pos = list(SH_freq.index).index(SH_cons_type)
                specificity = SH_cons_freq  # Z => SH
                sensitivity = 1 - AS_freq.iloc[cons_pos,i]  # not Z => AS
                accuracy = (sensitivity*self.AS_size + specificity*self.SH_size)/(self.AS_size + self.SH_size)
                tp = self.AS_size * sensitivity # not Z => AS
                fp = self.SH_size * (1 - SH_freq.iloc[cons_pos,i]) # not Z => SH
                tn = self.SH_size * specificity # Z => SH
                fn = self.AS_size * AS_freq.iloc[cons_pos,i] # Z => AS
                MCC = ((tp * tn) - (fp * fn))/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
                if tp == 1 or fp == 0 or fn ==0 or tn == 1:
                    p_value = 0
                else:					
                    table = np.array([[tp, fp], [fn, tn]]) # AS and SH have same contingency table
                    p_value = stats.chi2_contingency(table)[1]
                
                pos.append(i+1)
                sens.append(sensitivity * 100)
                spec.append(specificity * 100)
                acc.append(accuracy * 100)
                mcc.append(MCC)
                pvalue.append(p_value)
                key = self._one_to_three(SH_cons_type) if analysis=='amino' else SH_cons_type
                rule.append(f'not {key}=>AS, {key}=>SH')
            
            # Rule 3: [X ==> AS, Z ==> SH]
            if AS_cons_type != SH_cons_type and '-' not in [AS_cons_type, SH_cons_type]:
                #sensitivity, specificity = AS_cons_freq, SH_cons_freq
                cons_posX = list(AS_freq.index).index(AS_cons_type)
                cons_posZ = list(SH_freq.index).index(SH_cons_type)
                #accuracy = (sensitivity*self.AS_size + specificity*self.SH_size)/(self.AS_size + self.SH_size)
                tpX = AS_cons_freq * self.AS_size # X => AS
                fpX = SH_freq.iloc[cons_posX,i] * self.SH_size  # X => SH
                tnX = (1 - SH_freq.iloc[cons_posX,i]) * self.SH_size  # not X => SH
                fnX = self.AS_size * (1 - AS_freq.iloc[cons_posX,i]) # not X => AS
                
                tpZ = self.AS_size * (1 - AS_freq.iloc[cons_posZ,i]) # not Z => AS
                fpZ = self.SH_size * (1 - SH_freq.iloc[cons_posZ,i]) # not Z => SH
                tnZ = SH_cons_freq * self.SH_size # Y => SH
                fnZ = AS_freq.iloc[cons_posZ,i] * self.AS_size # Y => AS
                
                tp, fp, tn, fn = tpX + tpZ, fpX + fpZ, tnX + tnZ, fnX + fnZ
                MCC = ((tp * tn) - (fp * fn))/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
                if tp == 1 or fp == 0 or fn ==0 or tn == 1:
                    p_value = 0
                else:					
                    AS_table = np.array([[tp, fp], [fn, tn]])
                    p_value = stats.chi2_contingency(AS_table)[1]
                
                pos.append(i+1)
                sens.append(tp/(tp + fn) * 100)
                spec.append(tn/(tn + fp) * 100)
                acc.append((tn + tp)/(tn + tp + fp + fn) * 100)
                mcc.append(MCC)
                pvalue.append(p_value)
                keyX = self._one_to_three(AS_cons_type) if analysis=='amino' else AS_cons_type
                keyZ = self._one_to_three(SH_cons_type) if analysis=='amino' else SH_cons_type
                rule.append(f'{keyX}=>AS, {keyZ}=>SH')
                
        store = pd.DataFrame([pos, rule, sens, spec, acc, mcc, pvalue]).transpose()
        store.columns = ['tre_pos', 'rule', 'sens', 'spec', 'acc', 'mcc', 'pvalue']
        return store
            
    
    def site_plot(self, site, savefig=False, savepath=None):
        '''Plot bar graphs of amino acid composition for site.'''
        AS_comp = self.AS_freq.iloc[:20,site-1]*100
        SH_comp = self.SH_freq.iloc[:20,site-1]*100
        
        lw = 1.0   # Width of bar edge
        w = 0.25   #  Width of bar
        fnt = 'Arial'
        ticks_font = {'fontname':fnt, 'size':'20'}
        label_font = {'family':fnt, 'size':'22'}
        title_font = {'family':fnt, 'size':'24'}
        legend_font = {'family':'Arial', 'size':'18'}
        legend_label = ['AS', 'SH']
        plt.rcParams['grid.alpha'] = 0.5
        
        X = np.arange(len(AS_comp))
        
        out_AS = plt.bar(X-0.5*w, AS_comp, color='blue', width=w, linewidth=lw,
                       edgecolor='black')
        out_SH = plt.bar(X+0.5*w, SH_comp, color='red', width=w, linewidth=lw,
                       edgecolor='black')
        
        plt.yticks(**ticks_font)
        plt.xticks(X, AS_comp.index, rotation=0, **ticks_font)
        #plt.grid(True, linestyle='--')
        plt.ylabel('Frequency (%)', **label_font)
        plt.title(f'POS{site}', **title_font)
        pltout = [x[0] for x in [out_AS, out_SH]]
        plt.legend(pltout, legend_label, frameon=1, numpoints=1, shadow=1, loc='best', 
                   prop=legend_font)
        plt.tight_layout()
        
        if savefig:
           plt.savefig(f'{savepath}/pos{site}.pdf')
        plt.show()






# Distance between each residue and all glycosyl subsites
#============================================================#
if __name__ == '__main__':
	# Get pdb data (1mw)
	structure_id = '1mw0'
	filename = 'fasta/1mw0.pdb'
	parser=PDBParser(PERMISSIVE=1)
	structure = parser.get_structure(structure_id,filename)
	model = structure[0]
	chain = model['A']
	reslist = chain.get_list()


	def distance(x,y):
			'''Return the euclidean distance between 2 3D vectors.'''
			return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)
 

	def atom_distance(x,y):
		'''Returns the closest distance between two objects (x and y), i.e. the distance
		between the closest atoms in x and y.'''
		[x_atoms, y_atoms] = [Selection.unfold_entities(obj, 'A') for obj in [x,y]]
		distances = []
		for xatom in x_atoms:
			for yatom in y_atoms:
				distances.append(distance(xatom.get_coord(), yatom.get_coord()))
		return min(distances)


	reslist = Selection.unfold_entities(structure,'R')  # all residues
	ligand_ids = [('H_GLC', x, ' ') for x in range(629,636)] # ids for 7 glycosyl residues in BGC
	ligand_res = [chain[x] for x in ligand_ids]    # residue objects of 7 glycosyl residues in BGC
	prot_res = reslist[:628]   # protein residues


	# Calculate distances between closest atoms of protein residues and glycosyl residues
	store = []
	for prot in prot_res:
		p_store = []
		for lig in ligand_res:
			p_store.append(atom_distance(prot,lig))
		store.append(p_store)
	store = pd.DataFrame(store)
	store.index = np.array(store.index)+1
	store.columns = ['-1', '+1', '+2', '+3', '+4', '+5', '+6']


	# Closest subsite
	min_dist = []
	closest_res = []
	for i in range(len(store.index)):
		distances = list(store.iloc[i,:])
		min_dist.append(min(distances))
		closest_res.append(store.columns[distances.index(min(distances))])
	dist_store = pd.DataFrame([closest_res, min_dist]).transpose()
	dist_store.index = store.index
	dist_store.columns = ['closest_subsite', 'distance']
	dist_store.to_csv('results_final/residue_distances.csv')






# Derive classification rules using the GH13MSA class
#=========================================================#
if __name__ == '__main__':
    ASmsa = 'fasta/GH13_positions_only/AS_cat.fasta'
    SHmsa = 'fasta/GH13_positions_only/SH_cat.fasta'
    GH13MSA = GH13MSA(ASmsa, SHmsa)
    GH13MSA.get_freq(include_gaps=True)
    rules_amino = GH13MSA.get_rules(analysis='amino')
    rules_amino['closest_subsite'] = [dist_store['closest_subsite'][x] for x in rules_amino.tre_pos]
    rules_amino['dist_subsite'] = [dist_store['distance'][x] for x in rules_amino.tre_pos]
    rules_type = GH13MSA.get_rules(analysis='type')
    rules_type['closest_subsite'] = [dist_store['closest_subsite'][x] for x in rules_type.tre_pos]
    rules_type['dist_subsite'] = [dist_store['distance'][x] for x in rules_type.tre_pos]
    rules = rules_amino.append(rules_type, ignore_index=True)
    
    # Save rules
    rules_amino.to_csv('results_final/rules/rules_amino.csv')
    rules_type.to_csv('results_final/rules/rules_type.csv')
    rules.to_csv('results_final/rules/rules_all.csv')
    
    
    # Get consensus sequences
    GH13MSA.get_consensus_sequences()
    consensus_AS = GH13MSA.consensus_AS
    consensus_SH = GH13MSA.consensus_SH
    bioinf.combine_fasta(['AS consensus', 'SH consensus'],[consensus_AS, consensus_SH],
                         'fasta/GH13_positions_only/consensus.fasta')



