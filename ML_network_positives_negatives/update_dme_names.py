# written to re-name the files for training and testing
# written 11-26-19 JLW


import pickle, os, csv

allf = [f for f in os.listdir('.') if '.txt' in f]
allf.remove('drugs_to_dmes_false_positives.txt')
allf.remove('drugs_to_dmes_true_positive.txt')

for f in allf:
	old_name = f
	new_name = 'dme_'+old_name.lower()
	cmd = 'mv %s %s'%(old_name, new_name)
	print(cmd)
	os.system(cmd)



