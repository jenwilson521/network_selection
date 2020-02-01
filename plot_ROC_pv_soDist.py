# Written to copy ROC code for pvalues
# and add results from sub-optimal distance
# written 1-28-20 JLW

import pickle,os,csv,matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.integrate import simps
from numpy import trapz

# calculate results from sub-optimal distances
# data from label extraction for doing counting
drugs_by_dme = pickle.load(open('data/tpfp_drugs_by_dme.pkl','rb'))
dmes_by_drug = pickle.load(open('data/tpfp_dmes_by_drug.pkl','rb'))
fpdmes_by_drug = pickle.load(open('data/tpfp_fpdmes_by_drug.pkl','rb'))
all_dmes = [k for k in drugs_by_dme.keys()]

positives = [(dme,dbid) for (dme,dbid_list) in drugs_by_dme.items() for dbid in dbid_list]
negatives = [(dme,dbid) for (dbid,dme_list) in fpdmes_by_drug.items() for dme in dme_list]

so_rdir = '/Users/jenniferwilson/Documents/pathfx_data_update/results/analyze_so_dists/'
all_dist = ['0.82','0.83','0.86','0.87','0.88','0.89','0.9','0.99']
roc_data = []
for sdist in all_dist:
	print(sdist)
	fp_to_tox_f = 'drugs_fp_to_tox_XXX.pkl'.replace('XXX',sdist)
	fp_to_tox = pickle.load(open(os.path.join(so_rdir,fp_to_tox_f),'rb'))
	false_pos = [(dme,dbid) for (dme,dbid_list) in fp_to_tox.items() for dbid in dbid_list]
	true_negatives = set(negatives).difference(set(false_pos))

	tp_to_tox_f = 'drugs_matched_to_tox_XXX.pkl'.replace('XXX',sdist)
	tp_to_tox = pickle.load(open(os.path.join(so_rdir,tp_to_tox_f),'rb'))
	true_pos = [(dme,dbid) for (dme,dbid_list) in tp_to_tox.items() for dbid in dbid_list]

	tpr = len(true_pos)/float(len(positives))
	tnr = len(true_negatives)/float(len(negatives))
	fpr = 1-tnr
	roc_data.append((fpr,tpr))
print("AUROC P-value: "+str(area))


# load results from p-value
pv_roc_values = pickle.load(open('pvalue_roc_values.pkl','rb'))

# plot ROC curve
fig,ax = plt.subplots()
(x,y) = zip(*pv_roc_values)
area2 = trapz(y,x)
print("AUROC P-value: "+str(area2))
ax.plot(x,y,linewidth=4.0,label='P-value')
(x,y) = zip(*roc_data)
ax.plot(x,y,linewidth=4.0,label='Distance')
area2 = trapz(y,x)
print("AUROC Distance: "+str(area2))

ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),linestyle=':',linewidth=4.0,color='k')
ax.set_xlabel('false positive rate',fontsize=12)
ax.set_ylabel('true positive rate',fontsize=12)
ax.set_yticks([0,0.5,1.0])
ax.set_xticks([0,0.5,1.0])
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('pvalue_ROC_dist.png',format='png')
