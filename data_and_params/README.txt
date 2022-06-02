alpha0_meds.pkl: alpha0 medians from obtain_alpha0.py, 
	1k patient variables are simulated first for every setting, 
	for every set of patient variables, alpha was computed 100 times, 
	since the process is random, even with the same alpha, different exposure prevalences are obtained,
	the file contains a dictionary with keys a, b, c ... corresponding to the variable simulation settings
	for every key, values are lists of alpha0 medians 
	entries correspong to 0.01, 0.02, 0.04 and 0.08 prevalences