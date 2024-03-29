# written 9-29-20 to save and plot
# the feature importance scores, JLW

import pickle,os,csv,matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import pandas as pd

rdir = 'log_reg' 
outf = os.path.join(rdir,"logistic_regression_all_feature_importance.xlsx")
allf = [f for f in os.listdir(rdir) if 'feat_imp_scores_092920.pkl' in f]
writer = pd.ExcelWriter(outf) 
for fif in allf:
	dme = fif.replace('_feat_imp_scores_092920.pkl','')
	print(dme)
	fi = pickle.load(open(os.path.join(rdir,fif),'rb'))
	df=pd.DataFrame(fi,columns=["FeatureImportance","Gene"])
	df = df[["Gene","FeatureImportance"]].sort_values("FeatureImportance",ascending=False)
	df.to_excel(writer,sheet_name = dme)
	fig_width = len(fi)/5

	fig,ax = plt.subplots(figsize=(fig_width,4))
	sorted_fi = sorted(fi,key = lambda x: x[0],reverse=True)
	(bar_vals,bar_labels) = zip(*sorted_fi)
	bar_inds = [i for (i,x) in enumerate(bar_vals)]
	ax.bar(bar_inds,bar_vals)
	ax.set_xticks(bar_inds)
	ax.set_xticklabels(bar_labels)
	plt.xticks(rotation=90)
	ax.set_ylabel("Feature importance")
	ax.set_xlabel("Network proteins")
	plot_title = dme.replace("_"," ").capitalize()
	ax.set_title(plot_title)
	plt.subplots_adjust(left=0.2, bottom=0.3)
	plt.savefig(os.path.join(rdir,"feature_importance_bar_"+dme+".png"),format="png")
writer.save()
#fif ='tardive_dyskinesia_feat_imp_scores_092920.pkl'
