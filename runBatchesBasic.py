#!/bin/python3

###########################################################################################
### Runs Arbor simulations of a single current-based synapse that undergoes early- and  ###
### late-phase plasticity. Averages over batches, each with a certain number of trials. ###
###########################################################################################

### Copyright 2022-2023 Jannik Luboeinski
### licensed under Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)
### Contact: mail[at]jlubo.net

#########################################################
### Imports and initialization ###

import os
import json
import time
import arborNetworkConsolidation as anc
from outputUtilities import setDataPathPrefix
from averageFileColumnsAdvanced import averageFileColumns

#########################################################
# runSimulationBatches
# Runs a number of batches with a certain number of trials for a given protocol;
# subsequently averages over trials and over batches
# - protocol: name of the protocol to be used (according '*.json' configuration, file must exist)
# - num_batches: number of batches
# - trials_per_batch: number of trials per batch
# - only_average [optional]: if True, only do averaging over batches
def runSimulationBatches(protocol, num_batches, trials_per_batch, only_average = False):
	config_file = f"config_smallnet2_{protocol}.json" # file containing the parameter configuration
	data_root = f"./arbor_data_{protocol}_{num_batches}x{trials_per_batch}"
	data_path_averaged = f"{data_root}/averaged_traces"

	if not only_average:

		# for each batch, run a certain number of trials and then average the data traces over time
		for batch in range(num_batches):
			batch_name = str(batch + 1)
			data_path_batch = os.path.join(data_root, batch_name)

			os.makedirs(data_path_batch) # create data directory and intermediates; an error is thrown if the directory exists already
			if not os.path.exists(data_path_averaged):
				os.mkdir(data_path_averaged) # create directory for averaged data

			# simulate trials
			for trial in range(trials_per_batch):
				print("--------------------------------------------")
				print(f"Batch {batch_name}, trial {trial + 1}:")
					
				# load configuration from JSON file
				config = json.load(open(config_file, "r"))
				config["simulation"]["sample_curr"] = 1

				# run simulation
				setDataPathPrefix(os.path.join(data_path_batch, f"trial_{trial}_data_"))
				recipe = anc.arborNetworkConsolidation(config)

			# average traces over time
			# columns: 1: Time, 2: V(0), 4: h(1,0), 5: z(1,0), 6: Ca(1,0), 8: p^C(0)
			averageFileColumns(f'{data_path_averaged}/{batch_name}.txt', data_path_batch, 'data', [], 'traces.txt', [2,4,5,6,8], skip_first_line=True, col_sep=' ')

	# average mean and variance over batches; also cf. Kloeden and Platen (1995)
	# columns: 1: Time, 2: mean of V(0), 3: std. dev. of V(0), 4: mean of h(1,0), 5: std. dev. of h(1,0), 
	#          6: mean of z(1,0), 7: std. dev. of z(1,0), 8: mean of Ca(1,0), 9: std. dev. of Ca(1,0),
	#          10: mean of p^C(0), 11: std. dev. of p^C(0)
	averageFileColumns(f'{data_root}/meta_mean_averaged_traces.txt', data_root, 'averaged_traces', [], '.txt', [2,4,6,8,10], skip_first_line=False, col_sep=' ') # mean averaging
	averageFileColumns(f'{data_root}/meta_stdev_averaged_traces.txt', data_root, 'averaged_traces', [], '.txt', [3,5,7,9,11], skip_first_line=False, col_sep=' ') # std.dev. averaging

#########################################################
### Basic early- and late-phase dynamics
if __name__ == "__main__":
	runSimulationBatches("basic_early", 10, 100)
	#runSimulationBatches("basic_early", 10, 100, only_average = True)
	runSimulationBatches("basic_late", 10, 10)
	#runSimulationBatches("basic_late", 10, 10, only_average = True)

