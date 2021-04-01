# Written to copy ROC code for pvalues
# and add results from sub-optimal distance
# written 1-28-20 JLW
# added CSI 1-31-20

import pickle,os,csv,matplotlib, statistics
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.integrate import simps
from numpy import trapz
import pandas as pd

# run through ROC data for each DME
csi_roc_dir = 'ML_network_positives_negatives/log_reg/' 
alf = [f for f in os.listdir(csi_roc_dir) if '_ylabels_scores_072720.pkl' in f] # tp, fp, threshlds from scipy model.fit
csi_roc = dict([(f.replace('_ylabels_scores_072720.pkl',''),os.path.join(csi_roc_dir,f)) for f in alf])
csi_dmes = [k for k in csi_roc.keys()]

# merge all roc data for DMEs with sufficient evidence
print('gathering roc data for csi analysis')
all_roc = [(y_ts,y_score) for (dme,rocf) in csi_roc.items() for (y_ts,y_score) in pickle.load(open(rocf,'rb'))]
min_score = min([x[1] for x in all_roc])
max_score = max([x[1] for x in all_roc])
step_size = (max_score-min_score)/100
num_pos = len([x for x in all_roc if x[0]==1])
num_neg = len([x for x in all_roc if x[0]==0])
csi_roc_data = []
all_precision = []
for i in np.arange(min_score,max_score,step_size):
	positives = [x for x in all_roc if x[1]>=i]
	tp = float(len([x for x in positives if x[0]==1]))
	fp = float(len([x for x in positives if x[0]==0]))
	negatives = [x for x in all_roc if x[1]<i]
	tn = float(len([x for x in negatives if x[0]==0]))
	fn = float(len([x for x in negatives if x[0]==1]))
	fpr = fp/(fp+tn)
	tpr = tp/(tp+fn)
	csi_roc_data.append((fpr,tpr))

	prec_val = tp/(tp+fp)
	all_precision.append(prec_val)

print("Average precision: ")
print(statistics.mean(all_precision))

# also plot ROC curves for individual DMEs for supplment
# plot ROC values against no. postives/ no. negatives
true_positives_dbid = pickle.load(open('true_positives_dbid.pkl','rb')) # all positive examples
drugs_fp_to_tox = pickle.load(open('false_positives_dbid.pkl','rb')) # all negative examples
print('Making ROC curve with individual DMEs')
fig,ax = plt.subplots()
num_dmes = len(csi_roc)
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_dmes))))
dme_roc_vs_num_inputs = []
for (dme,rocf) in sorted(csi_roc.items(),key=lambda x:x[0]):
	dme_data = []
	dme_roc_zip = pickle.load(open(rocf,'rb'))
	dme_roc = [x for x in dme_roc_zip]
	min_score = min([x[1] for x in dme_roc])
	max_score = max([x[1] for x in dme_roc])
	step_size = (max_score-min_score)/100
	test_curve = []
	dme_prec = []
	dme_recall = []
	dme_fscore = []
	for i in np.arange(min_score,max_score,step_size):
		positives = [x for x in dme_roc if x[1]>=i]
		tp = float(len([x for x in positives if x[0]==1]))
		fp = float(len([x for x in positives if x[0]==0]))
		negatives = [x for x in dme_roc if x[1]<i]
		tn = float(len([x for x in negatives if x[0]==0]))
		fn = float(len([x for x in negatives if x[0]==1]))
		fpr = fp/(fp+tn)
		tpr = tp/(tp+fn)			
		dme_data.append((fpr,tpr))
		test_curve.append((i,fpr,tpr))
		prec_val = tp/(tp+fp)
		dme_prec.append(prec_val)
		dme_recall.append(tpr)
		if prec_val > 0 or tpr > 0:
			dme_fscore.append((2*prec_val*tpr)/(prec_val+tpr))
		else:
			dme_fscore.append(0)

	(x,y) = zip(*dme_data)
	ax.plot(x,y,label=dme)
	pval_area = -trapz(y,x)
	print('Average precision: ')
	avg_prec =statistics.mean(dme_prec) 
	print(avg_prec)
	avg_recall =statistics.mean(dme_recall) 
	avg_fscore = statistics.mean(dme_fscore)

	dme_cap = dme.capitalize().replace('_',' ') # change syntax to look up in other dictionary
	total_tp = len(true_positives_dbid[dme_cap])
	total_fp = len(drugs_fp_to_tox[dme_cap])
	dme_roc_vs_num_inputs.append({'name':dme,'ROC value':pval_area,'Total TP':total_tp,'Total FP':total_fp,'AvgPrec':avg_prec,'AvgRecall':avg_recall,'AvgFScore':avg_fscore})
	
	
# format figure for supplement
ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),linestyle=':',linewidth=4.0,color='k')
ax.set_xlabel('false positive rate',fontsize=14)
ax.set_ylabel('true positive rate',fontsize=14)
ax.set_yticks([0,0.5,1.0])
ax.set_xticks([0,0.5,1.0])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
ax.legend(bbox_to_anchor=(1.,1.),prop={"size":10},)#ncol=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.subplots_adjust(right = 0.55555,bottom = 0.15, left = 0.15)
plt.savefig('indiv_dme_ROC_092120.png',format="png") 
		
# save and report individual DME ROCs vs. number of input variables 
dme_roc_vs_inputs_df = pd.DataFrame(dme_roc_vs_num_inputs).set_index('name')
dme_roc_vs_inputs_df['PosToNegRatio'] = dme_roc_vs_inputs_df['Total TP']/dme_roc_vs_inputs_df['Total FP']
dme_roc_vs_inputs_df.to_excel('DME_individal_ROC_values_input_counts.xls')

# create plots of values to visualize
fig,ax_arr = plt.subplots(1,3,sharey=True)
for (i,(x_var,ax)) in enumerate(zip(['Total TP','Total FP','PosToNegRatio'],ax_arr)):
	dme_roc_vs_inputs_df.plot.scatter(x = x_var,y = 'ROC value',ax=ax)
	if i == 0:
		ax.set_ylabel('ROC Value')
	ax.set_xlabel(x_var)
	ax.set_ylim([0,1])
plt.savefig('DME_ROC_vs_input_nums.png',format='png')

			

# method for mapping all dmes to csi_dmes
def check_csi_dme(raw_dme):
	look_up_dme = raw_dme.lower().replace(' ','_')
	if look_up_dme in csi_dmes:
		return True
	else:
		return False

## data from label extraction for doing counting
drugs_by_dme = pickle.load(open('data/tpfp_drugs_by_dme.pkl','rb'))
dmes_by_drug = pickle.load(open('data/tpfp_dmes_by_drug.pkl','rb'))
fpdmes_by_drug = pickle.load(open('data/tpfp_fpdmes_by_drug.pkl','rb'))

# start with all data, subset for dmes where we could perform logistic regression
all_dmes = [k for k in drugs_by_dme.keys()]
positives = [(dme,dbid) for (dme,dbid_list) in drugs_by_dme.items() for dbid in dbid_list if check_csi_dme(dme)]
negatives = [(dme,dbid) for (dbid,dme_list) in fpdmes_by_drug.items() for dme in dme_list if check_csi_dme(dme)]

## calculate results from sub-optimal distances
print('gathering data for sub-optimal distance')
so_rdir = 'analyze_so_dists/'
all_dist = ['0.82','0.83','0.86','0.87','0.88','0.89','0.9','0.99']
sodist_roc_data = []
for sdist in all_dist:
	print(sdist)
	fp_to_tox_f = 'drugs_fp_to_tox_XXX.pkl'.replace('XXX',sdist)
	fp_to_tox = pickle.load(open(os.path.join(so_rdir,fp_to_tox_f),'rb'))
	false_pos = [(dme,dbid) for (dme,dbid_list) in fp_to_tox.items() for dbid in dbid_list if check_csi_dme(dme)]
	true_negatives = set(negatives).difference(set(false_pos))

	tp_to_tox_f = 'drugs_matched_to_tox_XXX.pkl'.replace('XXX',sdist)
	tp_to_tox = pickle.load(open(os.path.join(so_rdir,tp_to_tox_f),'rb'))
	true_pos = [(dme,dbid) for (dme,dbid_list) in tp_to_tox.items() for dbid in dbid_list if check_csi_dme(dme)]

	tpr = len(true_pos)/float(len(positives))
	tnr = len(true_negatives)/float(len(negatives))
	fpr = 1-tnr
	sodist_roc_data.append((fpr,tpr))

# load the p-value data
tp_dme_drug_pvalue = pickle.load(open('tp_dme_drug_pvalue.pkl','rb'))
fp_dme_drug_pvalue = pickle.load(open('fp_dme_drug_pvalue.pkl','rb'))
all_pvals = [x[2] for x in tp_dme_drug_pvalue] + [x[2] for x in fp_dme_drug_pvalue]
min_pv = min(all_pvals)
max_pv = max(all_pvals)
step_size = (max_pv-min_pv)/float(100)
pval_roc = []
for i in np.arange(min_pv,max_pv,step_size):
	tp_l = [(dme,dbid,pv) for (dme,dbid,pv) in tp_dme_drug_pvalue if pv <=i and check_csi_dme(dme)]
	tp = float(len(set(tp_l)))
	fp_l = [(dme,dbid,pv) for (dme,dbid,pv) in fp_dme_drug_pvalue if pv <=i and check_csi_dme(dme)]
	fp = float(len(set(fp_l)))
	tn_l = [(dme,dbid,pv) for (dme,dbid,pv) in fp_dme_drug_pvalue if pv >i and check_csi_dme(dme)]
	tn = float(len(set(tn_l)))
	fn_l = [(dme,dbid,pv) for (dme,dbid,pv) in tp_dme_drug_pvalue if pv >i and check_csi_dme(dme)]
	fn = float(len(set(fn_l)))
	fpr = fp/(fp+tn)
	tpr = tp/(tp+fn)
	pval_roc.append((fpr,tpr))
	

# plot the data and area using trapz method
fig,ax = plt.subplots()
(x,y) = zip(*csi_roc_data)
csi_area = -trapz(y,x)
ax.plot(x,y,linewidth=6.0,linestyle='-',alpha=0.75,color='green',label='CSI area: '+"{0:.2f}".format(csi_area))
(x,y) = zip(*sodist_roc_data)
sodist_area = -trapz(y,x)
ax.plot(x,y,linewidth=6.0,linestyle='-',alpha=0.75,color='blue',label='SOdist area: '+"{0:.2f}".format(sodist_area))
(x,y) = zip(*pval_roc)
pval_area = trapz(y,x)
ax.plot(x,y,linewidth=6.0,linestyle='-',alpha=0.75,color='orange',label='P-value area: '+"{0:.2f}".format(pval_area))

ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),linestyle=':',linewidth=4.0,color='k')
ax.set_xlabel('false positive rate',fontsize=14)
ax.set_ylabel('true positive rate',fontsize=14)
ax.set_yticks([0,0.5,1.0])
ax.set_xticks([0,0.5,1.0])
for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
ax.legend(bbox_to_anchor=(1.,1.),prop={"size":14},)#ncol=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.subplots_adjust(right = 0.55555,bottom = 0.15, left = 0.15)
# plt.savefig('test_merged_ROC_072720.png',format="png")
plt.savefig('merged_ROC_072720.png',format="png")




## load results from p-value
#pv_roc_values = pickle.load(open('pvalue_roc_values.pkl','rb'))
#
## plot ROC curve
#fig,ax = plt.subplots(figsize=(9,5))
#(x,y) = zip(*pv_roc_values)
#area2 = trapz(y,x)
#print("AUROC P-value: "+str(area2))
##ax.plot(x,y,linewidth=2.0,label='P-value')
#(x,y) = zip(*roc_data)
##ax.plot(x,y,linewidth=2.0,label='Distance')
#area2 = trapz(y,x)
#print("AUROC Distance: "+str(area2))


