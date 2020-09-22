# Written to copy ROC code for pvalues
# and add results from sub-optimal distance
# written 1-28-20 JLW
# added CSI 1-31-20

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

so_rdir = 'analyze_so_dists/'
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


# load results from p-value
pv_roc_values = pickle.load(open('pvalue_roc_values.pkl','rb'))

# plot ROC curve
fig,ax = plt.subplots(figsize=(9,5))
(x,y) = zip(*pv_roc_values)
area2 = trapz(y,x)
print("AUROC P-value: "+str(area2))
#ax.plot(x,y,linewidth=2.0,label='P-value')
(x,y) = zip(*roc_data)
#ax.plot(x,y,linewidth=2.0,label='Distance')
area2 = trapz(y,x)
print("AUROC Distance: "+str(area2))

# run through ROC data for each DME
csi_roc_dir = 'ML_network_positives_negatives/log_reg/' 
alf = [f for f in os.listdir(csi_roc_dir) if '_roc_data.pkl' in f]
csi_roc = dict([(f.replace('_roc_data.pkl',''),os.path.join(csi_roc_dir,f)) for f in alf])

alf = [f for f in os.listdir(csi_roc_dir) if '_ts_roc_auc_score.pkl' in f]
csi_auc_ts = dict([(f.replace('_ts_roc_auc_score.pkl',''),os.path.join(csi_roc_dir,f)) for f in alf]) 

dme_single = 'pancreatitis'
for (i,(dme,roc_data_f)) in enumerate(csi_roc.items()):
	palpha = (1.0/len(csi_roc)*i)
	print(dme)
	if dme not in csi_auc_ts:
		continue
	roc_data = pickle.load(open(roc_data_f,'rb'))
	(x,y) = zip(*roc_data)
	area2 = trapz(y,x)
	print('AUROC,trapz: '+str(area2))
	skl_auroc_f = csi_auc_ts[dme]
	skl_auroc = pickle.load(open(skl_auroc_f,'rb'))
	print('AUROC,sklearn: '+str(skl_auroc))
	#if area2>0.7 and area2 <1.0:
	if area2>0.7 and area2 <1.0 and dme==dme_single:
		lbl = dme+" \nAUROC: "+"{0:.2f}".format(area2)
		ax.plot(x,y,linewidth=6.0,linestyle='-',label=lbl,alpha=0.6,color='green')#color='blueviolet',alpha=palpha)

ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),linestyle=':',linewidth=4.0,color='k')
ax.set_xlabel('false positive rate',fontsize=20)
ax.set_ylabel('true positive rate',fontsize=20)
ax.set_yticks([0,0.5,1.0])
ax.set_xticks([0,0.5,1.0])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
ax.legend(bbox_to_anchor=(1.,1.),prop={"size":16},)#ncol=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.subplots_adjust(right = 0.6,bottom = 0.15, left = 0.15)
# plt.savefig('improve_top_methods_ROC_dist.png',format='png')
plt.savefig('improve_top_methods_ROC_dist_'+dme_single+'.png',format='png')
