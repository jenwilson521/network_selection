# written to look at the numbers of positive and negative examples
# for each DME
# written 3-19-20 JLW

import pickle,os,csv
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

allf = [f for f in os.listdir('.') if 'dme_' in f and '.txt' in f]

num_pos = {}
num_fp = {}
for f in allf:
        dme = f.replace('dme_','').replace('.txt','')
        print(dme)
	d = [l.strip().split('\t') for l in open(f,'rU').readlines()]
	tp = len([x for x in d if x[1] =='positive'])
	fp = len([x for x in d if x[1] =='negative'])
	num_pos[dme] = tp
	num_fp[dme] = fp


barWidth = 0.25
r1 = np.arange(len(num_pos))
r2 = [x + barWidth for x in r1]
x_tick_lbls = sorted([k for k in num_pos.keys()],key = lambda x:x[0])

fig,ax = plt.subplots()
tp = [num_pos[dme] for dme in x_tick_lbls]
fp = [num_fp[dme] for dme in x_tick_lbls]
ax.bar(r1,tp,label='TP',width=barWidth)
ax.bar(r2,fp,label='FP',width=barWidth)
ax.set_ylabel('No. of Examples')
ax.set_xlabel('DME case')
plt.xticks([r + barWidth for r in range(len(x_tick_lbls))], x_tick_lbls,rotation='vertical')
plt.legend()
plt.subplots_adjust(bottom=0.4)
plt.savefig('Num_tp_fp_by_dme.png',format='png')


