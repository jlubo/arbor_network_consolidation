##################################################################################################
### Script that averages over trials of Q and MI data, employing 'extractParamsQMIfromSpikes.py' #
##################################################################################################
### Copyright 2023 Jannik Luboeinski
### licensed under Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)
### Contact: mail@jlubo.net

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./analysis')
import extractParamsQMIfromSpikes

# ---------------------------------------------------------------------------------------
# parameters
Nl_exc = 40
Nl_inh = 20
col_sep = "\x20"
params_Q_MI_file = "Params_Q_MI.txt"
averaged_Q_MI_file = "data_Q_MI.txt"

# ---------------------------------------------------------------------------------------
# extract Q and MI measures from data
fout_1 = open(params_Q_MI_file, "w")
extractParamsQMIfromSpikes.extractRecursion(".", fout_1, col_sep, Nl_exc**2, Nl_inh**2)
fout_1.close()

# ---------------------------------------------------------------------------------------
# loop over pattern sizes
fout_2 = open(averaged_Q_MI_file, "w")
for pattern_size in [100, 150, 200, 250, 300]:

		# ---------------------------------------------------------------------------------------
		# select rows for specific pattern size and diffusivity value
		df_all_trials = pd.read_table(params_Q_MI_file, header=None, sep=col_sep, engine='python')
		df_selected_trials = df_all_trials[df_all_trials[1] == pattern_size]
		num_trials = len(df_selected_trials.index)

		# ---------------------------------------------------------------------------------------
		# compute mean and std. dev. across trials (for firing rates, Q, MI)
		nu_exc_mean = df_selected_trials.loc[:,7].mean(axis=0)
		nu_exc_sd = df_selected_trials.loc[:,7].std(axis=0)
		nu_as_mean = df_selected_trials.loc[:,9].mean(axis=0)
		nu_as_sd = df_selected_trials.loc[:,9].std(axis=0)
		nu_ans_mean = df_selected_trials.loc[:,11].mean(axis=0)
		nu_ans_sd = df_selected_trials.loc[:,11].std(axis=0)
		nu_ctrl_mean = df_selected_trials.loc[:,13].mean(axis=0)
		nu_ctrl_sd = df_selected_trials.loc[:,13].std(axis=0)
		nu_inh_mean = df_selected_trials.loc[:,15].mean(axis=0)
		nu_inh_sd = df_selected_trials.loc[:,15].std(axis=0)
		Q_mean = df_selected_trials.loc[:,17].mean(axis=0)
		Q_sd = df_selected_trials.loc[:,17].std(axis=0)
		MI_mean = df_selected_trials.loc[:,19].mean(axis=0)
		MI_sd = df_selected_trials.loc[:,19].std(axis=0)
		selfMIL_mean = df_selected_trials.loc[:,20].mean(axis=0)
		selfMIL_sd = df_selected_trials.loc[:,20].std(axis=0)
		print(f"Pattern size {pattern_size}:")
		print(f"  nu_exc = {nu_exc_mean} +- {nu_exc_sd}")
		print(f"  nu_as = {nu_as_mean} +- {nu_as_sd}")
		print(f"  nu_ans = {nu_ans_mean} +- {nu_ans_sd}")
		print(f"  nu_ctrl = {nu_ctrl_mean} +- {nu_ctrl_sd}")
		print(f"  nu_inh = {nu_inh_mean} +- {nu_inh_sd}")
		print(f"  Q = {Q_mean} +- {Q_sd}")
		print(f"  MI = {MI_mean} +- {MI_sd}")
		print(f"  selfMIL = {selfMIL_mean} +- {selfMIL_sd}")
		fout_2.write(str(pattern_size) + col_sep +
			         str(Q_mean) + col_sep + str(Q_sd) + col_sep +
			         str(MI_mean) + col_sep + str(MI_sd) + col_sep +
			         str(selfMIL_mean) + col_sep + str(selfMIL_sd) + col_sep +
		             str(num_trials) + "\n")
fout_2.close()

