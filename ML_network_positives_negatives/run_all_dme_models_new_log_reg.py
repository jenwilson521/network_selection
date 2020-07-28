# written to run all DMEs through logistic regression and
# save the results
# written 11-27-19 JLW, updated 12-5-19 to use nested validation
# re-written 7-20-20 JLW to redo logistic regression with selected
# dmes for which there are enough examples

import pickle,os,csv,matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np

model_type = 'log_reg' # 'dec_tree' # 'rand_for'
print('\n\nRUNNING: '+model_type+'\n\n')

allf = [f for f in os.listdir('.') if 'dme_' in f and '.txt' in f]

for f in allf:
	dme = f.replace('dme_','').replace('.txt','')
	print('\n\n\n'+dme)
	# check if there are sufficient positive/negative labels
	d = [l.strip().split('\t') for l in open(f,'rU').readlines()]
	tp = len([x for x in d if x[1] =='positive'])
	tn = len([x for x in d if x[1] =='negative'])
	if tp > 10 and tn > 10:
		cmd = 'python all_pathways.py %s -m %s -n %s'%(f,model_type,dme)
		print(cmd)
		os.system(cmd)
	else:
		print('skipped')


