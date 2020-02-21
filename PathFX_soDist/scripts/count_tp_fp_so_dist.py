# written to gather data for a ROC
# curve using so distances in the PathFX 
# algorithm on tp, fp drug-DME associations
# written 1-28-20 JLW

import pickle, os, csv
from collections import defaultdict

rdir = '../results/'
dat_dir = '../data/'
rscs_dir = '../rscs/'
save_dir = os.path.join(rdir,'analyze_so_dists/')

dintf = os.path.join(rscs_dir,'drug_intome_targets_orig.pkl')
dint = pickle.load(open(dintf,'rb')) # to find number of drug targets

# data from label extraction for doing counting
drugs_by_dme = pickle.load(open('../data/tpfp_drugs_by_dme.pkl','rb'))
dmes_by_drug = pickle.load(open('../data/tpfp_dmes_by_drug.pkl','rb'))
fpdmes_by_drug = pickle.load(open('../data/tpfp_fpdmes_by_drug.pkl','rb'))

# data and methods for matching phenotype names
dme_syns = pickle.load(open(os.path.join(dat_dir,'synonyms.pkl'),'rb'))
def strip_string(s):
        return s.replace(',','').replace('(','').replace(')','')

def exact_overlap(t,p): # check if all tox words are contained in phenotype
	t_words = [x.lower() for x in t.split(' ')]
	p_words = [x.lower() for x in strip_string(p).split(' ')]
	t_len = float(len(t_words))
	if len(set(t_words).intersection(set(p_words))) == t_len:
		return True
	else:
		return False

def check_synonyms(t,p):
	match = False
	if dme_syns[t] != []:
		syns = dme_syns[t]
		syns = [x for x in syns if x!= '']
		for s in syns:
			if exact_overlap(s,p):
				match = True
	return match

# find a dictionary of all network files at that directory
def get_netf_dic(rd,froot):
	nd = dict([(f.replace(froot,''),os.path.join(ssd,f)) for [ssd,igdir,flist] in os.walk(rd) for f in flist if froot in f])
	return nd


# get all network files at each distance
all_sdir_list = [f for f in os.walk('../results/')][0][1]
so_dirs = [(sd.replace('alldb_so_dist_',''),os.path.join(rdir,sd)) for sd in all_sdir_list if 'alldb_so_dist' in sd]
all_dist = [sod for (sod,sdirname) in so_dirs]

# find all expected p-value distances by threshold value 
ex_path = '../results/so_dist_XXXrandom_networks/so_dist_XXX_expected_pvalue_summary.pkl'
exp_pv_dic = dict([(so_dist,ex_path.replace('XXX',so_dist)) for so_dist in all_dist])

## FOR PROTOTYPING ##
## so_dirs = [('0.88', '../results/alldb_so_dist_0.88'), ('0.89', '../results/alldb_so_dist_0.89'),('0.99', '../results/alldb_so_dist_0.99')]
# so_dirs.remove(('0.84','../results/alldb_so_dist_0.84'))
for (so_dist,sd) in so_dirs:
	print(so_dist)
	net_dic = get_netf_dic(sd,'_merged_neighborhood__assoc_table_.txt')
	exp_pvf = exp_pv_dic[so_dist]
	print(exp_pvf)
	exp_pv = pickle.load(open(exp_pvf,'rb'),encoding="bytes")
	
	drugs_matched_to_tox = defaultdict(set)
	true_positives_dbid = defaultdict(set)
	drugs_fp_to_tox = defaultdict(set)

	
	for (dbid,asf) in net_dic.items():
		if dbid not in dint:
			continue
		targs = dint[dbid]
		nt = len(targs)
		drug_dmes = dmes_by_drug[dbid] # DMEs extracted from drug label
		fp_dmes = fpdmes_by_drug[dbid]	# DMEs not listed on the drug label
		dR = csv.DictReader(open(asf,'r'),delimiter='\t')

		for row in dR: # get each phenotype for that drug
			[rank,ph,cui,asin,asii,BH,genes] = [row['rank'],row['phenotype'],row['cui'],row['assoc in neigh'],row['assoc in intom'],float(row['Benjamini-Hochberg']),row['genes']]
			if cui in exp_pv[nt]:
				exp_BH = exp_pv[nt][cui]
			else:
				exp_BH = 1
			if BH < exp_BH: # only look for matches with the significant phenotypes
				for tox in drug_dmes:
					if exact_overlap(tox,ph) or check_synonyms(tox,ph):
						drugs_matched_to_tox[tox].add(dbid)
						#out_data = [dname,dbid,tox,ph,cui,str(exp_BH),genes,'\n']
						#of1.write('\t'.join(out_data))
						true_positives_dbid[tox].add((dbid,ph,genes)) # save the dbid-phenotype pairs

				for tox in fp_dmes:
					if exact_overlap(tox,ph) or check_synonyms(tox,ph):
						drugs_fp_to_tox[tox].add(dbid)
						#out_data = [dname,dbid,tox,ph,cui,str(exp_BH),genes,'\n']
						#of2.write('\t'.join(out_data))

	pickle.dump(drugs_matched_to_tox,open(os.path.join(save_dir,'drugs_matched_to_tox_'+so_dist+'.pkl'),'wb'))
	pickle.dump(true_positives_dbid,open(os.path.join(save_dir,'true_positives_dbid_'+so_dist+'.pkl'),'wb'))
	pickle.dump(drugs_fp_to_tox,open(os.path.join(save_dir,'drugs_fp_to_tox_'+so_dist+'.pkl'),'wb'))
