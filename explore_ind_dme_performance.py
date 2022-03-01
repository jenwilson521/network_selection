 # written to investigate the inputs to each DME further
# to understand differences in performance across each DME
# written 4-1-21 JLW

import pickle,os,csv
import pandas as pd
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

# look at input files to count columns and measure matrix sparsity
data_dir = './ML_network_positives_negatives/'
allf = [f for f in os.listdir(data_dir) if 'dme_' in f and '.txt' in f]
input_file_dic = dict([(f.replace('dme_','').replace('.txt',''),os.path.join(data_dir,f)) for f in allf])

dme_to_num_columns = {}
dme_to_singleton_columns = {}
dme_frac_shared_propo = {}

def get_proportions(df,col_names):
	proportions = [(c,(df[c].sum())/df.shape[0]) for c in col_names]
	return proportions

def count_similar_representation(ca,cb):
	if abs(ca-cb) <=0.1: # require 20% similiarity
		return True
	else:
		return False

for (dme,dme_file) in input_file_dic.items():
	print(dme)
	dme_df = pd.read_table(dme_file).set_index('name')
	gene_cols = [x for x in dme_df.columns if x!='label']
	num_col = len(gene_cols) # total number of genes considered
	dme_to_num_columns[dme] = num_col

	# fraction of gene columns with only one entry
	singletons = [c for c in gene_cols if c!='label' and dme_df[c].sum()==1.0]
	fract_sing = len(singletons)/float(num_col)
	dme_to_singleton_columns[dme] = fract_sing

	# count number of genes with similar representation in the positive and negative groups
	pos_df = dme_df[dme_df['label'] == 'positive']
	neg_df = dme_df[dme_df['label'] == 'negative']

	pos_prop = dict(get_proportions(pos_df,gene_cols))
	neg_prop = dict(get_proportions(neg_df,gene_cols))

	sim_rep_genes = [c for c in gene_cols if count_similar_representation(pos_prop[c],neg_prop[c])]
	frac_simil = float(len(sim_rep_genes))/num_col
	dme_frac_shared_propo[dme] = frac_simil

# use dictionaries to modify and export dataframe
dme_roc_vs_inputs_df = pd.read_excel('DME_individual_ROC_values_input_counts.xls')
dme_roc_vs_inputs_df["Fractionpos/total"] = dme_roc_vs_inputs_df['Total TP']/(dme_roc_vs_inputs_df['Total TP']+dme_roc_vs_inputs_df['Total FP'])
dme_roc_vs_inputs_df['TotalNumGenes'] = dme_roc_vs_inputs_df['name'].map(dme_to_num_columns)
dme_roc_vs_inputs_df['NumSingletonGenes'] = dme_roc_vs_inputs_df['name'].map(dme_to_singleton_columns)
dme_roc_vs_inputs_df['FractionSharedGenes'] = dme_roc_vs_inputs_df['name'].map(dme_frac_shared_propo)

dme_roc_vs_inputs_df.set_index('name')
dme_roc_vs_inputs_df.to_excel('DME_individual_ROC_values_input_counts_expanded.xls',index=False)


# create plots of values to visualize
fig,ax_arr = plt.subplots(4,7,sharey=True,figsize = (12,8))
for (j,x_var) in enumerate(['Total TP','Total FP','PosToNegRatio','Fractionpos/total','TotalNumGenes','NumSingletonGenes','FractionSharedGenes']):
	for (i,y_var) in enumerate(['AvgFScore','AvgPrec','AvgRecall','ROC value']):
		ax = ax_arr[i][j]
		dme_roc_vs_inputs_df.plot.scatter(x = x_var,y = y_var,ax=ax,color='k',alpha=0.5)
		(x,y) = (dme_roc_vs_inputs_df[x_var], dme_roc_vs_inputs_df[y_var]) 
		b, m = polyfit(x,y, 1)
		ax.plot(x, b + m * x, '-',color='red',alpha=0.75)
		if j == 0:
			ax.set_ylabel(y_var)
#		if i == 3:
#			ax.set_xlabel(x_var)
		ax.set_ylim([0,1])

plt.savefig('DME_perf_vs_input_nums_exp.png',format='png')

# exploratory
#rel_prop = [(c,abs(pos_prop[c]-neg_prop[c])) for c in gene_cols if count_similar_representation(pos_prop[c],neg_prop[c])]

