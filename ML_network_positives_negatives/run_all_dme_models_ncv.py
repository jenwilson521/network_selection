# written to run all DMEs through logistic regression and
# save the results
# written 11-27-19 JLW, updated 12-5-19 to use nested validation

import pickle,os,csv,matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

allf = [f for f in os.listdir('.') if 'dme_' in f and '.txt' in f]
all_dmes = sorted([x.replace('dme_','').replace('.txt','') for x in allf])
dme_ind = dict([(d,i) for (i,d) in enumerate(all_dmes)]) # index for model type

# model_type = 'rand_for'# 'log_reg' # 'dec_tree' # 'rand_for'
model_types = ['log_reg','dec_tree','rand_for']
m_i = dict([(m,i) for (i,m) in enumerate(model_types)]) # index for model type

val_type ='nestedcv' 
# for model_type in model_types:
for model_type in ['log_reg','dec_tree']:
	print('\n\nRUNNING: '+model_type+'\n\n')
	for f in allf:
		dme = f.replace('dme_','').replace('.txt','')
		print(dme)
		# check for if there are sufficient positive cases?
		d = [l.strip().split('\t') for l in open(f,'rU').readlines()]
		tp = len([x for x in d if x[1] =='positive'])
		tn = len([x for x in d if x[1] =='negative'])
	#	print(str(tp)+'/'+str(tn)+'/'+str(len(d)-1))
		if tp > 10 and tn > 10:
			cmd = 'python all_pathways_with_validation.py %s -m %s -n %s -v %s -s %s'%(f,model_type,dme,'nest',val_type)
			print(cmd)
#			os.system(cmd)

		else:
			print('skipped')

## plot all scores on a bar chart
allf = [f for f in os.listdir('nestedcv_nest') if 'rcv_nest_validation_res.pkl' in f]
scts = ['f1', 'roc', 'accuracy']
s_i = dict([(m,i) for (i,m) in enumerate(scts)]) # index for score types
all_means = np.zeros((len(all_dmes),len(model_types),len(scts))) 
all_std = np.zeros((len(all_dmes),len(model_types),len(scts)))
for f in allf:
	model_type = [x for x in model_types if x in f][0]
	c = m_i[model_type]
	fshort = f.replace(model_type,'')
	dme = fshort.replace('__rcv_nest_validation_res.pkl','')
	r = dme_ind[dme]
	res = pickle.load(open(os.path.join('nestedcv_nest',f),'rb'))
	[mn,std] = [dict(res['mean']),dict(res['std'])]
	for st in scts:
		z = s_i[st]
		all_means[r,c,z] = mn[st]
		all_std[r,c,z] = std[st]
	
fig,ax_arr = plt.subplots(3,1,sharex=True,figsize=(5,7.5))
for (st,si) in s_i.items():
	ax = ax_arr[si]
	for (mt,mi) in m_i.items():
		score_vals = all_means[:,mi,si]
		score_err = all_std[:,mi,si]
		ax.errorbar(range(len(all_dmes)),score_vals,yerr = score_err,label=mt,fmt='o',alpha=0.5)
		print(st+'\t'+mt+':'+str(np.mean([x for x in score_vals if x>0])))
	ax.set_title(st)
	ax.set_ylim([0, 1])
	ax.set_xticks(range(len(all_dmes)))
	ax.set_yticks([0.5,0.75,1.0])
ax.set_xticklabels(all_dmes,rotation='vertical')
ax_arr[0].legend(loc='upper center',bbox_to_anchor=(0.5,1.5),ncol=len(scts))
plt.subplots_adjust(bottom=0.3)
plt.savefig('compare_NCV_scores_all_models.png',format='png')
plt.close()



## plot feature importance
outf = 'log_reg_coeff_vs_dec_tree_feat_imp.xlsx'
writer = pd.ExcelWriter(outf)
ord_col = ['name','dec tree feat imp','dec tree err','log reg coeff','log reg err']
allf = [f for f in allf if 'log_reg' in f]
for f in allf:
	dme = f.split("_log_reg")[0]
	lr = pickle.load(open(os.path.join('nestedcv_nest',f),'rb'))
	dt = pickle.load(open(os.path.join('nestedcv_nest',f.replace("log_reg", "dec_tree")),'rb'))

	lr_fi = dict([(nm,(fval,ferr)) for (nm,fval,ferr) in lr['features_importance']])
	dt_fi = dict([(nm,(fval,ferr)) for (nm,fval,ferr) in dt['features_importance']])
	xlabel = sorted([k for k in lr_fi.keys()])

	lr_coef_ = np.zeros((2,len(xlabel)))
	dt_coef_ = np.zeros((2,len(xlabel)))
	row_data = []
	# make sure ordering of features is the same
	for (i,xl) in enumerate(xlabel):
		rd = {}
		lr_coef_[0][i] = lr_fi[xl][0] 
		lr_coef_[1][i] = lr_fi[xl][1]
		dt_coef_[0][i] = dt_fi[xl][0]
		dt_coef_[1][i] = dt_fi[xl][1]
		rd['name'] = xl
		rd['log reg coeff'] = lr_fi[xl][0]
		rd['log reg err'] = lr_fi[xl][1]
		rd['dec tree feat imp'] = dt_fi[xl][0]
		rd['dec tree err'] = dt_fi[xl][1]
		row_data.append(rd)
	df = pd.DataFrame(row_data)
	df = df[ord_col].sort_values(by=['dec tree feat imp'],ascending=False)
	df.to_excel(writer,sheet_name = dme)

#	lr_coef_ = [[l[1] for l in lr_fi],
#				[l[2] for l in lr_fi]]
#	dt_fi = dt['features_importance']
#	dt_feat_imp = [[d[1] for d in dt_fi],
#				[d[2] for d in dt_fi]]

	# Setting the positions and width for bars
	pos = list(range(len(xlabel))) 
	width = 0.25
	# Plotting the bars
	fig,ax_arr = plt.subplots(2,1,sharex=True,figsize=(5,7.5))

	for i, (value, color, l, ylabel) in enumerate([(lr_coef_, "green", "log_reg", "coefficent value"), (dt_coef_, "red", "det_tree", "feature importance")]):
		ax = ax_arr[i]
		rects = ax.bar(pos, value[0], width, 
					   alpha=0.5, color=color, yerr = value[1]) ## to remove error bars just comment yerr
		ax.set_ylabel(ylabel)

		if not i:
			ax.axhline(0, color='black', linewidth=0.1)
		ax.set_title(l)
		ax.set_xticks(pos)
		if i:
			ax.set_xticklabels(xlabel,rotation='vertical')

	plt.tight_layout()
	fig.savefig("histogram_" + dme + ".png")
	plt.close(fig)
writer.close()

