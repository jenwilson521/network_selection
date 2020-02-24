# written to run all of DrugBank through
# each threshold of PathFX
# written 1/21/20 JLW

import pickle, os, csv
from collections import defaultdict

# run all of DrugBank???
dbf = '../rscs/pathfx_mp_dbid2name.pkl'
db2name = pickle.load(open(dbf,'rb'))

# Just run those that are in the DME analysis
#drugs_by_dme = defaultdict(list)
#dmes_by_drug = defaultdict(list)
#unique_dmes = [k for k in drugs_by_dme.keys()]
# fpdmes_by_drug = defaultdict(list)

unique_drugs = pickle.load(open('../data/tpfp_unique_drugs.pkl','rb'))
print(len(unique_drugs))

# Get all versions of PathFX, start with a base call to the algorithm
base_cmd = "python PFX_NAME -d DBID -a ANAME"
all_pfx = [f for f in os.listdir('.') if 'phenotype_enrichment_pathway_so_dist_' in f]
# print(all_pfx)
# all_pfx = ['phenotype_enrichment_pathway_so_dist_0.9.py','phenotype_enrichment_pathway_so_dist_0.84.py']

for pfx in all_pfx:
	analysis_name = 'allDB_' + pfx.replace('phenotype_enrichment_pathway_','').replace('.py','')
	int_cmd = base_cmd.replace('PFX_NAME',pfx).replace('ANAME',analysis_name)
	for drugbank_id in unique_drugs:
#	for drugbank_id in ['DB00331','DB01268','DB01221']:
		cmd = int_cmd.replace('DBID',drugbank_id)
		print('\n'+cmd)
		os.system(cmd)
