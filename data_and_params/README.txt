alpha0_dic.pkl: alpha0, every dictionary entry corresponds to one variable simulation setting
	the dictionary values are lists, every list entry corresponds to on prevalence
	The simulated prevalence is first averaged over 1000 datasets, then bisection is performed
beta0_exp_ls.pkl: For every setting a-f beta0_exp is determined using bisection
				such that the incidence is 0.1
beta_exp_ls.pkl: For every setting a-f and every prevalence beta_exp is determined using bisection
				such that the risk difference is 0.02, the process is the same as with
				alpha0 and beta0_exp