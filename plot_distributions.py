# written to look at how true positives
# and true negatives separate using three methods
# p-value, distance, decision tree?
# written 11-25-19 JLW

import pickle, os, csv, matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from collections import defaultdict

tpf = 'true_positives_dbid.pkl'
fpf = 'false_positives_dbid.pkl'

tp = pickle.load(open(tpf,'rb'))
fp = pickle.load(open(fpf,'rb'))


# methods for phenotype matching
dme_syns = pickle.load(open(os.path.join(data_dir,'synonyms.pkl'),'rb'))
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

# load drug-DME data from drug labels
f = os.path.join(data_dir,'Drugs_labeled_for_AEs.txt') #Each column header is a DME, and the rows are drugs associated with the DME
dR = csv.DictReader(open(f,'r'),delimiter='\t')

# gather starting data
unique_drugs = set()
drugs_by_dme = defaultdict(list)
dmes_by_drugs = defaultdict(list)
for r in dR:
        for (dme,drug) in r.items():
                if drug != '':
                        drugs_by_dme[dme].append(drug.lower())
                        unique_drugs.add(drug.lower())
                        dmes_by_drugs[drug.lower()].append(dme)
print('extracted drug-dme relationships')

# create set of false positives
fpdmes_by_drugs = defaultdict(list)
unique_dmes = [k for k in drugs_by_dme.keys()]
for (drug,dme_list) in dmes_by_drugs.items():
        fp_dmes = list(set(unique_dmes).difference(set(dme_list))) # get list of dmes NOT on the drug label
        fpdmes_by_drugs[drug.lower()] = fp_dmes

# map to DrugBank IDs
un_in_DB = [n2db[n.lower()] for n in unique_drugs if n.lower() in n2db] #1136 drugs

# check how many are in drug bank and modeled
db_with_targf = os.path.join(rscs_dir,'drugBank_with_at_least_one_intom_targ.pkl')
db_with_targ = pickle.load(open(db_with_targf,'rb'))
un_with_targ = [n for n in un_in_DB if n in db_with_targ] #997 drugs
un_with_targ_dname = [db2n[d].lower() for d in un_with_targ] #997 drugs
num_drug_analyzed = len(un_with_targ)
un_w_t_in_dme = [d for d in un_with_targ if db2n[d].lower() in dmes_by_drugs] # drug with targets and dmes
print('loaded drug bank mappings')

# dictionary to look up drug network results
asfdir = os.path.join(data_dir,'all_drugbank_network_association_files/')
asf = [f for f in os.listdir(asfdir)]
asf_dic = dict([(f.split('_')[0],f) for f in asf if 'merged_neighborhood__assoc_table_.txt' in f])
print('loaded association files')

# other data needed for analysis
# expected p-values for each dme, and number of drug targets
back_phmeds_f = os.path.join(rscs_dir,'back_phmeds.pkl')
back_phmeds = pickle.load(open(back_phmeds_f,'rb')) # to find expected p-value for phenotype
dintf = os.path.join(rscs_dir,'drug_intome_targets.pkl')
dint = pickle.load(open(dintf,'rb')) # to find number of drug targets

# run analysis, store data for calculating specificity
drugs_matched_to_tox = defaultdict(set)
true_positives_dbid = defaultdict(set)
drugs_fp_to_tox = defaultdict(set)
of1 = open('drugs_to_dmes_true_positive.txt','w')
hdr = ['Drug Name','DrugBankID','Label DME','PathFX Phenotype','PathFX Phen. CUI','Corrected P-value','Network genes assoc to phen','\n']
of1.write('\t'.join(hdr))
of2 = open('drugs_to_dmes_false_positives.txt','w')
hdr = ['Drug Name','DrugBankID','False Positive DME','PathFX Phenotype','PathFX Phen. CUI','Corrected P-value','Network genes assoc to phen','\n']
of2.write('\t'.join(hdr))
for dbid in un_w_t_in_dme:
        print('analyzing drug: '+dbid)
        asf = os.path.join(asfdir,asf_dic[dbid])
        targs = dint[dbid]
        nt = len(targs)
        dname = db2n[dbid].lower()
        drug_dmes = dmes_by_drugs[dname]
        fp_dmes = fpdmes_by_drugs[dname]
        dR = csv.DictReader(open(asf,'r'),delimiter='\t')
        for row in dR: # get each phenotype for that drug
                [rank,ph,cui,asin,asii,BH,genes] = [row['rank'],row['phenotype'],row['cui'],row['assoc in neigh'],row['assoc in intom'],float(row['Benjamini-Hochberg']),row['genes']]
                if cui in back_phmeds[nt]:
                        exp_BH = back_phmeds[nt][cui]
                else:
                        exp_BH = 1
                if BH < exp_BH: # only look for matches with the significant phenotypes
                        for tox in drug_dmes:
                                if exact_overlap(tox,ph) or check_synonyms(tox,ph):
                                        drugs_matched_to_tox[tox].add(dname)
                                        out_data = [dname,dbid,tox,ph,cui,str(exp_BH),genes,'\n']
                                        of1.write('\t'.join(out_data))
                                        true_positives_dbid[tox].add((dbid,ph,genes)) # save the dbid-phenotype pairs

                        for tox in fp_dmes:
                                if exact_overlap(tox,ph) or check_synonyms(tox,ph):
                                        drugs_fp_to_tox[tox].add(dname)
                                        out_data = [dname,dbid,tox,ph,cui,str(exp_BH),genes,'\n']
                                        of2.write('\t'.join(out_data))
of1.close()
of2.close()
pickle.dump(true_positives_dbid,open(os.path.join(data_dir,'true_positives_dbid.pkl'),'wb'))
pickle.dump(drugs_fp_to_tox,open(os.path.join(data_dir,'false_positives_dbid.pkl'),'wb'))


