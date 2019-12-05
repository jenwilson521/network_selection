# written to run all DMEs through logistic regression and
# save the results
# written 11-27-19 JLW, updated 12-5-19 to use nested validation

import pickle,os,csv,matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

model_type = 'rand_for'# 'log_reg' # 'dec_tree' # 'rand_for'
print('\n\nRUNNING: '+model_type+'\n\n')

allf = [f for f in os.listdir('.') if 'dme_' in f and '.txt' in f]

for f in allf:
	dme = f.replace('dme_','').replace('.txt','')
	cmd = 'python all_pathways.py %s -m %s -n %s'%(f,model_type,dme)
	print(cmd)
	os.system(cmd)

# plot all scores on a bar chart
res_dir = os.path.join('.',model_type)
all_res = [os.path.join(res_dir,f) for f in os.listdir(res_dir) if '.pkl' in f]
tr_acc = [f for f in all_res if '_tr_acc_score.pkl' in f]
#print(tr_acc)
tr_f1 = [f for f in all_res if '_tr_f1_score.pkl' in f]
tr_roc = [f for f in all_res if '_tr_roc_auc_score.pkl' in f]

ts_acc = [f for f in all_res if '_ts_acc_score.pkl' in f]
ts_f1 = [f for f in all_res if '_ts_f1_score.pkl' in f]
ts_roc = [f for f in all_res if '_ts_roc_auc_score.pkl' in f]

barWidth = 0.25
r1 = np.arange(len(tr_acc))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
x_tick_lbls = [os.path.split(f)[-1].replace('_tr_acc_score.pkl','') for f in sorted(tr_acc)]

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(10,10))
avgs = {}
for (flist,lbl,x_pos,clr) in [(tr_acc,'accuracy',r1,'gainsboro'),(tr_f1,'f1 score',r2,'grey'),(tr_roc,'roc score',r3,'black')]:
	vals = [pickle.load(open(f,'rb')) for f in sorted(flist)]
	ax1.bar(x_pos,vals,color=clr,width=barWidth,label=lbl)
	avgs[lbl] = '{:.2f}'.format(np.mean(vals))
avg_str = ' '.join([(k+': '+v) for (k,v) in avgs.items()])
ax1.set_title('training\n'+avg_str)
avgs = {}
for (flist,lbl,x_pos,clr) in [(ts_acc,'accuracy',r1,'gainsboro'),(ts_f1,'f1 score',r2,'grey'),(ts_roc,'roc score',r3,'black')]:
	vals = [pickle.load(open(f,'rb')) for f in sorted(flist)]
	ax2.bar(x_pos,vals,color=clr,width=barWidth,label=lbl)
	avgs[lbl] = '{:.2f}'.format(np.mean(vals))
avg_str = ' '.join([(k+': '+v) for (k,v) in avgs.items()])
ax2.set_title('testing\n'+avg_str)

# formatting
ticks = [0.0,0.5,0.75,0.9,1.0]
ax1.set_ylabel('score',fontsize=12)
ax1.set_ylim([0,1.0])
ax1.set_yticks(ticks)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.axhline(y=0.5, color='k', linestyle=':')
ax2.set_ylabel('score',fontsize=12)
ax2.set_ylim([0,1.0])
ax2.set_yticks(ticks)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.axhline(y=0.5, color='k', linestyle=':')

plt.xlabel('DME', fontweight='bold',fontsize=12)
plt.xticks([r + barWidth for r in range(len(x_tick_lbls))], x_tick_lbls,rotation='vertical')
plt.legend()
plt.subplots_adjust(bottom=0.4)
plt.savefig('compare_performance_'+model_type+'.png',format = 'png')

