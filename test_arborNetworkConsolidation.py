#!/bin/python3

# Tests for the arborNetworkConsolidation module

# Copyright 2022-2024 Jannik Luboeinski
# License: Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Contact: mail[at]jlubo.net

import numpy as np
import json
import os
import inspect
import pytest
import arborNetworkConsolidation as anc
from arbor import units as U
from outputUtilities import redText, cyanText, \
                            setDataPathPrefix, getDataPath

epsilon = 6e-7 # similar to default tolerance value used by pytest.approx()
setDataPathPrefix("tmp_data_")  # added "tmp_" for test results

###############################################################################
# Test the connectivity of ad-hoc generated networks of 2000 neurons
@pytest.mark.parametrize("config_file", 1*["config_defaultnet-like.json"])
def test_generated_connectivity(config_file):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	config["simulation"]["runtime"] = 0.1 # very short runtime (we are only interested in the initial connectivity)
	config["simulation"]["sample_gid_list"] = [ ] # no neuron probes
	config["simulation"]["sample_pre_list"] = [ ] # no synapse probes
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ('{config_file}')", platform = "CPU")

	# test the numbers of excitatory and inhibitory neurons
	assert recipe.exc_neurons_counter == config['populations']['N_exc']
	assert recipe.inh_neurons_counter == config['populations']['N_inh']

	# test if number of connections of the generated connectivity matrix does not deviate more than 5% from the expected value
	assert np.abs(recipe.p_c * (recipe.N_tot**2 - recipe.N_tot) - recipe.connections_counter) < 0.05 * recipe.p_c * (recipe.N_tot**2 - recipe.N_tot)

###############################################################################
# Test the connectivity of the pre-defined default network of 2000 neurons
@pytest.mark.parametrize("config_file", ["config_defaultnet.json"])
def test_defn_connectivity(config_file):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	config["simulation"]["runtime"] = 0.1 # very short runtime (we are only interested in the initial connectivity)
	config["simulation"]["sample_gid_list"] = [ ] # no neuron probes
	config["simulation"]["sample_pre_list"] = [ ] # no synapse probes
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation 
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ('{config_file}')", platform = "CPU")

	# test the number of connections of the pre-defined connectivity matrix
	assert recipe.connections_counter == 400012 

	# test for the existence of specific connections in the default connectivity matrix
	assert recipe.conn_matrix[68][6] == 1 # 6->68
	assert recipe.conn_matrix[68][16] == 1 # 16->68
	assert recipe.conn_matrix[0][53] == 1 # 53->0

	# find and test the internal connection numbers of the E->E connections 6->68 and 53->0 [...and *->0]
	inc_connections = np.sum(recipe.conn_matrix[68][0:recipe.N_exc-1], dtype=int) # number of incoming excitatory connections
	n = 0
	inc_connections2 = np.sum(recipe.conn_matrix[0][0:recipe.N_exc-1], dtype=int) # number of incoming excitatory connections
	m = 0
	inc_connections3 = np.sum(recipe.conn_matrix[862][0:recipe.N_exc-1], dtype=int) # number of incoming excitatory connections
	l = 0
	for i in range(recipe.N_exc):
		if recipe.conn_matrix[68][i] == 1:
			if i == 6:
				print("Connection 6->68: incoming synapse #" + str(inc_connections-n-1) + " of neuron 68.") # print the connection number used by Arbor
				assert inc_connections-n-1 == 153
			n += 1
		if recipe.conn_matrix[0][i] == 1:
			#print("Connection " + str(i) + "->0: incoming synapse #" + str(inc_connections2-m) + " of neuron 0.") # print the connection number used by Arbor
			if i == 53:
				print("Connection 53->0: incoming synapse #" + str(inc_connections2-m-1) + " of neuron 0.") # print the connection number used by Arbor
				assert inc_connections2-m-1 == 161
			m += 1
		if recipe.conn_matrix[862][i] == 1:
			#print("Connection " + str(i) + "->0: incoming synapse #" + str(inc_connections2-m) + " of neuron 0.") # print the connection number used by Arbor
			if i == 0:
				print("Connection 0->862: incoming synapse #" + str(inc_connections3-l-1) + " of neuron 862.") # print the connection number used by Arbor
				assert inc_connections3-l-1 == 155
			l += 1

###############################################################################	
# Test the response of the pre-defined default network when one neuron is stimulated to fire a single spike
@pytest.mark.parametrize("test_type, config_file", [("onespike_ee", "test_config_defaultnet_onespike_ee.json"), \
                                                    ("onespike_ei", "test_config_defaultnet_onespike_ei.json"), \
                                                    ("onespike_ie", "test_config_defaultnet_onespike_ie.json"), \
                                                    ("onespike_ii", "test_config_defaultnet_onespike_ii.json")])
def test_defn_onespike(test_type, config_file):

	#-----------------------------------------------------------------------------------
	# printValues
	# Prints the values of the main neuronal and synaptic measures at specified times
	# - data_stacked: array containing all the data
	# - tb_PSP_begin: timestep at which neuronal values shall be read
	# - tb_Ca_begin: timestep at which synaptic values shall be read
	# - dt: duration of one timestep
	def printValues(data_stacked, tb_PSP_begin, tb_Ca_begin, dt):
		print("t_PSP_begin =", tb_PSP_begin*dt) # start time of PSP caused by the first stimulus
		print("V(t_PSP_begin) =", data_stacked[tb_PSP_begin][1]) # membrane potential upon first stimulus
		print("I_stim(t_PSP_begin) =", data_stacked[tb_PSP_begin][2]) # stimulation current upon first stimulus

		print("t_Ca_begin =", tb_Ca_begin*dt) # start time of calcium increase caused by the first stimulus
		print("h(t_Ca_begin) =", data_stacked[tb_Ca_begin][3]) # early-phase weight upon first stimulus
		print("z(t_Ca_begin) =", data_stacked[tb_Ca_begin][4]) # late-phase weight upon first stimulus
		print("Ca(t_Ca_begin) =", data_stacked[tb_Ca_begin][5]) # calcium amount upon first stimulus
		print("p(t_Ca_begin) =", data_stacked[tb_Ca_begin][6]) # signal triggering protein synthesis upon first stimulus
		print("p(t_Ca_begin) =", data_stacked[tb_Ca_begin][7]) # protein concentration upon first stimulus

	# configuration (with explicit input but without learning and recall protocols)
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config["simulation"]["short_description"] # short description of the simulation
	dt = config["simulation"]["dt"]
	learn_prot = anc.completeProt(config["simulation"]["learn_protocol"])

	n_stim = len(learn_prot["explicit_input"]["stim_times"]) # the number of stimulus pulses
	config["simulation"]["sample_curr"] = 1 # retrieve Ornstein-Uhlenbeck stimulation current
	sample_gid = config["simulation"]["sample_gid_list"][0] # gid of the neuron to be probed (e.g., 68)
	sample_presyn_gid = config["simulation"]["sample_pre_list"][0] # gid of the presynaptic neuron for probing synapses (e.g., 6 for synapse 6->68)
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation 
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ({test_type}, '{config_file}')", platform = "CPU")

	# test the functional effects of a spike that is evoked at a determined time
	tb_before_last_spike = int((learn_prot["explicit_input"]["stim_times"][n_stim-1] + config["synapses"]["t_ax_delay"])/dt) # timestep right before the onset of the PSP caused by the last stimulus
	tb_before_last_Ca_increase = int((learn_prot["explicit_input"]["stim_times"][n_stim-1] + config["synapses"]["syn_exc_calcium_plasticity"]["t_Ca_delay"])/dt) # timestep right before the presynapse-induced calcium increase caused by the last stimulus
	data_stacked = np.loadtxt(getDataPath(s_desc, "traces.txt"))

	printValues(data_stacked, tb_before_last_spike, tb_before_last_Ca_increase+2, dt)

	assert data_stacked[tb_before_last_spike+2][2] == pytest.approx(0) # upon PSP onset: stimulation current equal zero (again)

	# specifics of the different tests
	if ("onespike" in test_type):
		assert data_stacked[tb_before_last_spike][1] == pytest.approx(config["neuron"]["V_rev"]) # before PSP: membrane potential equal V_rev
		if (test_type == "onespike_ee"):
			assert sample_gid == 68
			assert sample_presyn_gid == 6
			assert data_stacked[tb_before_last_spike+2][1] > config["neuron"]["V_rev"] + epsilon # upon PSP onset: membrane potential greater than V_rev
			assert data_stacked[tb_before_last_Ca_increase+2][5] \
				   == pytest.approx(config["synapses"]["syn_exc_calcium_plasticity"]["Ca_pre"], 1e-2) # calcium amount approx. equal to Ca_pre
		elif (test_type == "onespike_ei"):
			assert sample_gid == 1760
			assert data_stacked[tb_before_last_spike+2][1] > config["neuron"]["V_rev"] + epsilon # upon PSP onset: membrane potential greater than V_rev
		elif (test_type == "onespike_ie"):
			assert sample_gid == 17
			assert data_stacked[tb_before_last_spike+2][1] < config["neuron"]["V_rev"] - epsilon # upon PSP onset: membrane potential less than V_rev
		elif (test_type == "onespike_ii"):
			assert sample_gid == 1690
			assert data_stacked[tb_before_last_spike+2][1] < config["neuron"]["V_rev"] - epsilon # upon PSP onset: membrane potential less than V_rev
	else:
		raise ValueError("Unknown test type.")

###############################################################################	
# Test of the network simulation with background noise and with ad-hoc generated connectivity
@pytest.mark.parametrize("config_file", ["config_defaultnet.json"])
def test_defn_background_noise(config_file):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config['simulation']['short_description'] # short description of the simulation
	config['simulation']['runtime'] = 1 # use only short runtime of 1 second
	config['simulation']['learn_protocol'] = anc.completeProt({ }) # no learning protocol
	config['simulation']['recall_protocol'] = anc.completeProt({ }) # no recall protocol
	config['simulation']['sample_gid_list'] = [0] # probe neuron 0
	config['simulation']['sample_pre_list'] = [27] # probe synapse incoming from neuron 27
	config['simulation']['sample_curr'] = 2 # retrieve Ornstein-Uhlenbeck background noise current
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ('{config_file}')", platform = "CPU")

	# test the averaged noise current
	data_stacked = np.loadtxt(getDataPath(s_desc, "traces.txt"))
	mean_I_bg = np.mean(data_stacked[:,2])
	assert mean_I_bg == pytest.approx(config['simulation']['bg_protocol']['I_0'], 2) # actual mean of the noise current approx. equal to adjusted mean

	# test the numbers of excitatory and inhibitory neurons
	assert recipe.exc_neurons_counter == config['populations']['N_exc'] 
	assert recipe.inh_neurons_counter == config['populations']['N_inh']

	# test if number of connections of the generated connectivity matrix does not deviate more than 10% from the expected value
	assert np.abs(recipe.p_c * (recipe.N_tot**2 - recipe.N_tot) - recipe.connections_counter) < 0.1 * recipe.p_c * (recipe.N_tot**2 - recipe.N_tot)

###############################################################################	
# Testing a neuron that fires at maximal activity
@pytest.mark.parametrize("config_file", ["test_config_defaultnet_max_activity.json", "test_config_defaultnet_max_activity_no_conn.json"])
def test_defn_max_activity(config_file, num_spikes_max_activity):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config['simulation']['short_description'] # short description of the simulation
	if config["populations"]["p_c"] > epsilon: # in the case that there is connectivity
		config['simulation']['sample_gid_list'] = [862] # neuron to consider
		config['simulation']['sample_pre_list'] = [0] # synapse incoming from neuron 0
	else: # in the case that there is no connectivity
		config['simulation']['sample_gid_list'] = [862] # neuron to consider
		config['simulation']['sample_pre_list'] = [-1] # no synapse to consider
	config['simulation']['sample_curr'] = 1 # retrieve Ornstein-Uhlenbeck stimulation current
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ('{config_file}')", platform = "CPU")

	# get the spikes
	spike_data = np.loadtxt(getDataPath(s_desc, "spikes.txt")).transpose()
	spikes_0 = spike_data[0][spike_data[1] == 0] # spike times of neuron 0

	# test the number of spikes # TODO check why it is always a bit lower in the unconnected case!
	print("Number of spikes in neuron 0: ", len(spikes_0))
	assert len(spikes_0) == pytest.approx(1000*config['simulation']['runtime'] / (config['neuron']['t_ref']+config['simulation']['dt']), 0.1) # number of spikes approx. equal to t_stim/t_ref
	if config["populations"]["p_c"] > epsilon: # in the case that there is connectivity
		if num_spikes_max_activity['no_conn'] is not None: # if test without connectivity has run already
			assert num_spikes_max_activity['no_conn'] == len(spikes_0) # assert that connectivity does not matter (CROSS-TEST)
		else:
			num_spikes_max_activity['conn'] = len(spikes_0) # store the number of spikes
	else: # in the case that there is no connectivity
		assert len(spike_data[0]) == len(spikes_0) # assert that only neuron 0 spikes
		if num_spikes_max_activity['conn'] is not None: # if test with connectivity has run already
			assert len(spikes_0) == pytest.approx(num_spikes_max_activity['conn'], 0.1) # assert that connectivity does not matter (CROSS-TEST)
		else:
			num_spikes_max_activity['no_conn'] = len(spikes_0) # store the number of spikes

###############################################################################	
# Test of a neuron that is stimulated with a constant current (with and without relaxation after 100 ms)
@pytest.mark.parametrize("config_file", ["test_config_defaultnet_conv.json", "test_config_defaultnet_conv_relax.json"])
def test_def_const_conv_relax(config_file):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config['simulation']['short_description'] # short description of the simulation
	I_0 = 0.15 # constant input current (in nA)
	config['simulation']['sample_gid_list'] = [0] # probe neuron 0
	config['simulation']['sample_pre_list'] = [-1] # no synapse to consider
	config['simulation']['sample_curr'] = 0 # retrieve total current
	config['simulation']['bg_protocol']['I_0'] = I_0
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ('{config_file}')", platform = "CPU")

	# test the initial state
	data_stacked = np.loadtxt(getDataPath(s_desc, "traces.txt"))
	assert data_stacked[0][1] == pytest.approx(config['neuron']['V_rev'], 1e-4) # membrane potential equal to V_rev
	assert data_stacked[0][2] == pytest.approx(I_0) # total current equal to I_0

	# test the convergence
	dt = config["simulation"]["dt"]
	R_mem = config["neuron"]["R_leak"]
	tb_converged_1 = int(75/dt) # assuming convergence after 75 ms
	tb_converged_2 = int(175/dt) # assuming second convergence after 175 ms (during of "ONEPULSE" protocol plus 75 ms)
	assert data_stacked[tb_converged_1][1] == pytest.approx(config['neuron']['V_rev'] + R_mem*I_0, 1e-4) # depolarized membrane potential
	assert abs(data_stacked[tb_converged_2][2]) < 5e-4 # total current equal zero (again)

	# test the relaxation if there is no persistent background input
	if config['simulation']['bg_protocol']['scheme'] != "FULL":
		assert data_stacked[tb_converged_2][1] == pytest.approx(config['neuron']['V_rev'], 1e-4) # membrane potential equal V_rev (again)
		assert abs(data_stacked[tb_converged_2][2]) < 5e-4 # total current equal zero (again)

###############################################################################	
# Test the connectivity of a pre-defined network of 3 neurons
@pytest.mark.parametrize("network, config_file", [("smallnet", "config_smallnetX_non-det.json"), \
                                                  ("smallnet2", "config_smallnetX_non-det.json"), \
                                                  ("smallnet3", "config_smallnetX_non-det.json")])
def test_smallnetX_connectivity(network, config_file):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	config['simulation']['sample_gid_list'] = [ ] # no neuron probes
	config['simulation']['sample_pre_list'] = [ ] # no synapse probes
	config['simulation']['sample_curr'] = 0 # retrieve total current
	config['simulation']['runtime'] = 0.1 # very short runtime (we are only interested in the initial connectivity)
	config['simulation']['learn_protocol'] = anc.completeProt({ }) # no learning protocol
	config['simulation']['recall_protocol'] = anc.completeProt({ }) # no recall protocol
	config['populations']['conn_file'] = f"connections_{network}.txt" # connectivity matrix
	if network == "smallnet3":
		config['populations']['N_inh'] = 1
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ({network}, '{config_file}')", platform = "CPU")

	if network == "smallnet":
		# test the number of connections of the pre-defined connectivity matrix
		assert recipe.connections_counter == 9

		# test for the existence of specific connections in the connectivity matrix
		assert recipe.conn_matrix[1][0] == 1 # 0->1
		assert recipe.conn_matrix[2][0] == 1 # 0->2
		assert recipe.conn_matrix[3][0] == 1 # 0->3
		assert recipe.conn_matrix[0][1] == 0 # 1->0
		assert recipe.conn_matrix[2][1] == 1 # 1->2
		assert recipe.conn_matrix[3][1] == 1 # 1->3
		assert recipe.conn_matrix[0][2] == 0 # 2->0
		assert recipe.conn_matrix[1][2] == 1 # 2->1
		assert recipe.conn_matrix[3][2] == 1 # 2->3
		assert recipe.conn_matrix[0][3] == 0 # 3->0
		assert recipe.conn_matrix[1][3] == 1 # 3->1
		assert recipe.conn_matrix[2][3] == 1 # 3->2

	elif network == "smallnet2" or network == "smallnet3": # (smallnet3 extends smallnet2)
		# test for the existence of specific connections in the connectivity matrix
		assert recipe.conn_matrix[1][0] == 1 # 0->1
		assert recipe.conn_matrix[2][0] == 1 # 0->2
		assert recipe.conn_matrix[3][0] == 1 # 0->3
		assert recipe.conn_matrix[0][1] == 0 # 1->0
		assert recipe.conn_matrix[2][1] == 0 # 1->2
		assert recipe.conn_matrix[3][1] == 0 # 1->3
		assert recipe.conn_matrix[0][2] == 0 # 2->0
		assert recipe.conn_matrix[1][2] == 0 # 2->1
		assert recipe.conn_matrix[3][2] == 1 # 2->3
		assert recipe.conn_matrix[0][3] == 0 # 3->0
		assert recipe.conn_matrix[1][3] == 0 # 3->1
		assert recipe.conn_matrix[2][3] == 1 # 3->2

		if network == "smallnet2":
			# test the number of connections of the pre-defined connectivity matrix
			assert recipe.connections_counter == 5

		elif network == "smallnet3":
			# test the number of connections of the pre-defined connectivity matrix
			assert recipe.connections_counter == 9

			# test for the existence of specific connections in the connectivity matrix
			assert recipe.conn_matrix[4][2] == 1 # 2->4
			assert recipe.conn_matrix[4][3] == 1 # 3->4
			assert recipe.conn_matrix[2][4] == 1 # 4->2
			assert recipe.conn_matrix[3][4] == 1 # 4->3

###############################################################################	
# Test the response of a small network of 3 exc. neurons and up to 1 inh. neuron, where one exc. neuron is stimulated to fire at maximal activity
@pytest.mark.parametrize("network, config_file", [("smallnet", "config_smallnetX_non-det.json"), \
                                                  ("smallnet2", "config_smallnetX_non-det.json"), \
                                                  ("smallnet3", "config_smallnetX_non-det.json")])
def test_smallnetX_max_activity(network, config_file):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config['simulation']['short_description'] # short description of the simulation
	learn_prot = anc.completeProt(config["simulation"]["learn_protocol"])
	config['simulation']['sample_gid_list'] = [0] # neuron probes
	config['simulation']['sample_pre_list'] = [-1] # no synapse to consider
	config['simulation']['sample_curr'] = 0 # retrieve total current
	config['populations']['conn_file'] = f"connections_{network}.txt" # connectivity matrix
	if network == "smallnet3":
		config['populations']['N_inh'] = 1
	h_0 = config['synapses']['syn_exc_calcium_plasticity']['h_0']
	R_leak = config['neuron']['R_leak']
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ({network}, '{config_file}')", platform = "CPU")

	# test the initial state
	data_stacked = np.loadtxt(getDataPath(s_desc, "traces.txt"))
	assert data_stacked[0][1] == pytest.approx(config['neuron']['V_rev'], 1e-4) # membrane potential equal to V_rev
	assert data_stacked[0][2] == pytest.approx(learn_prot['N_stim']*h_0/R_leak) # total current equal to stimulation

	# get the spikes
	spike_data = np.loadtxt(getDataPath(s_desc, "spikes.txt")).transpose()
	spikes_0 = spike_data[0][spike_data[1] == 0] # spike times of neuron 0

	# test the number of spikes
	print("Number of spikes in neuron 0: ", len(spikes_0))
	assert len(spikes_0) == pytest.approx(1000*config['simulation']['runtime'] / config['neuron']['t_ref'], 0.15) # number of spikes approx. equal to t_stim/t_ref

###############################################################################	
# Test the response of a small network of 3 exc. neurons and up to 1 inh. neuron, where one exc. neuron is stimulated to fire a single spike
@pytest.mark.parametrize("network, config_file, platform",
                         [("smallnet2", "test_config_smallnet2_det_onespike.json", "CPU"),
                          ("smallnet2", "test_config_smallnet2_det_onespike.json", "GPU"),
                          ("smallnet3", "test_config_smallnet3_det_onespike.json", "CPU"),
                          ("smallnet3", "test_config_smallnet3_det_onespike.json", "GPU")])
def test_smallnetX_onespike(network, config_file, platform):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config['simulation']['short_description'] # short description of the simulation
	dt = config["simulation"]["dt"]
	learn_prot = anc.completeProt(config["simulation"]["learn_protocol"])
	if network == "smallnet2":
		config['simulation']['sample_gid_list'] = [2,3] # neurons to consider (3 is stimulated and projects to 2)
		config['simulation']['sample_pre_list'] = [3,2] # presynaptic neurons to consider for synapse probing
	elif network == "smallnet3":
		config['simulation']['sample_gid_list'] = [2,3,4] # neurons to consider (3 is stimulated and projects to 2 and 4)
		config['simulation']['sample_pre_list'] = [3,2,3] # presynaptic neurons to consider for synapse probing
	config['simulation']['sample_curr'] = 0 # retrieve total current
	config['populations']['conn_file'] = f"connections_{network}.txt" # connectivity matrix
	h_0 = config['synapses']['syn_exc_calcium_plasticity']['h_0']
	R_leak = config['neuron']['R_leak']
	w_ei = config['populations']['w_ei']
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ('{config_file}')", platform = platform)

	# test impact of the spike
	tb_before_first_spike = int((learn_prot["explicit_input"]["stim_times"][0] + config["synapses"]["t_ax_delay"])/dt) # timestep right before the onset of the PSP caused by the (first) stimulus
	tb_before_first_Ca_pre_increase = int((learn_prot["explicit_input"]["stim_times"][0] + config["synapses"]["syn_exc_calcium_plasticity"]["t_Ca_delay"])/dt) # timestep right before the presynapse-induced calcium increase caused by the (first) stimulus
	tb_before_first_Ca_post_increase = int((learn_prot["explicit_input"]["stim_times"][0])/dt) # timestep right before the psotsynapse-induced (backpropagating) calcium increase caused by the (first) stimulus
	data_stacked = np.loadtxt(getDataPath(s_desc, "traces.txt"))
	assert data_stacked[tb_before_first_spike][1] == pytest.approx(config["neuron"]["V_rev"]) # before PSP onset: membrane potential in neuron 2 equal V_rev
	assert data_stacked[tb_before_first_spike][2] == pytest.approx(h_0/R_leak, 1e-4) # before PSP onset: total current in neuron 2 approximately equal to intial E->E weight
	assert data_stacked[tb_before_first_spike+1][1] > config["neuron"]["V_rev"] + epsilon # upon PSP onset: membrane potential in neuron 2 greater than V_rev
	assert data_stacked[tb_before_first_Ca_post_increase+2][12] == pytest.approx(config['synapses']['syn_exc_calcium_plasticity']['Ca_post'], 0.01) # calcium equal to Ca_post
	assert data_stacked[tb_before_first_Ca_pre_increase+2][5] == pytest.approx(config['synapses']['syn_exc_calcium_plasticity']['Ca_pre'], 0.01) # calcium equal to Ca_pre
	if network == "smallnet3":
		assert data_stacked[tb_before_first_spike][15] == pytest.approx(config["neuron"]["V_rev"]) # before PSP onset: membrane potential in neuron 4 equal V_rev
		assert data_stacked[tb_before_first_spike][16] == pytest.approx(w_ei*h_0/R_leak, 1e-4) # before PSP onset: total current in neuron 4 approximately equal to E->I weight
		assert data_stacked[tb_before_first_spike+1][15] > config["neuron"]["V_rev"] + epsilon # upon PSP onset: membrane potential in neuron 4 greater than V_rev

	# get spikes & test the number of spikes
	spike_data = np.loadtxt(getDataPath(s_desc, "spikes.txt")).transpose()
	spikes_0 = spike_data[0][spike_data[1] == 0] # spike times of neuron 0
	spikes_1 = spike_data[0][spike_data[1] == 1] # spike times of neuron 1
	spikes_2 = spike_data[0][spike_data[1] == 2] # spike times of neuron 2
	spikes_3 = spike_data[0][spike_data[1] == 3] # spike times of neuron 3
	assert len(spikes_0) == 0 # zero spikes in neuron 0 expected
	assert len(spikes_1) == 0 # zero spikes in neuron 1 expected
	assert len(spikes_2) == 0 # zero spikes in neuron 2 expected
	assert len(spikes_3) == 1 # one spike in neuron 3 expected

###############################################################################	
# Test schedules of learning phase, consolidation phase, and recall phase, or background activity in a small network
@pytest.mark.parametrize("runtime, learn, recall, config_file, platform",
                         [(28820, True, True, "test_config_smallnet3_det_learn_8h-recall.json", "CPU"),
                          (28820, True, True, "test_config_smallnet3_det_learn_8h-recall.json", "GPU"),
                          (4, True, True, "test_config_smallnet3_det_learn_2s-recall.json", "CPU"),
                          (4, True, True, "test_config_smallnet3_det_learn_2s-recall.json", "GPU"),
                          (28820, True, False, "test_config_smallnet3_det_learn_8h-recall.json", "CPU"),
                          (28820, True, False, "test_config_smallnet3_det_learn_8h-recall.json", "GPU"),
                          (4, True, False, "test_config_smallnet3_det_learn_2s-recall.json", "CPU"),
                          (4, True, False, "test_config_smallnet3_det_learn_2s-recall.json", "GPU"),
                          (28820, False, True, "test_config_smallnet3_det_learn_8h-recall.json", "CPU"),
                          (28820, False, True, "test_config_smallnet3_det_learn_8h-recall.json", "GPU"),
                          (4, False, True, "test_config_smallnet3_det_learn_2s-recall.json", "CPU"),
                          (4, False, True, "test_config_smallnet3_det_learn_2s-recall.json", "GPU"),
                          (28820, False, False, "test_config_smallnet3_det_learn_8h-recall.json", "CPU"),
                          (28820, False, False, "test_config_smallnet3_det_learn_8h-recall.json", "GPU"),
                          (4, False, False, "test_config_smallnet3_det_learn_2s-recall.json", "CPU"),
                          (4, False, False, "test_config_smallnet3_det_learn_2s-recall.json", "GPU")])
def test_smallnet3_schedules(runtime, learn, recall, config_file, platform):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config['simulation']['short_description'] # short description of the simulation
	config['simulation']['runtime'] = runtime # runtime in seconds
	if learn:
		s_desc += ", learn"
	else:
		s_desc += ", no learn"
		config['simulation']['learn_protocol'] = anc.completeProt({ }) # no learning protocol
	if recall:
		s_desc += ", recall"
	else:
		s_desc += ", no recall"
		config['simulation']['recall_protocol'] = anc.completeProt({ }) # no recall protocol
	s_desc += f", {runtime}, {platform}"
	config['simulation']['short_description'] = s_desc
	config['simulation']['sample_gid_list'] = [1] # probe neuron 1
	config['simulation']['sample_pre_list'] = [0] # probe synapse incoming from neuron 0
	config['simulation']['sample_curr'] = 1 # retrieve Ornstein-Uhlenbeck stimulation current

	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ('{config_file}')", platform = platform)

	# test if the right schedule has been selected
	keyword = "Schedule: "
	schedule = None
	with open(getDataPath(s_desc, "log.txt"), "r") as f:
		for line in f:
			if line.find(keyword) == 0:
				schedule = line[len(keyword):].rstrip()
	print(f"{keyword}{schedule}")
	if learn and recall and runtime >= 25:
		assert schedule == "learn - consolidate - recall"
	elif learn and recall and runtime < 25:
		assert schedule == "learn+recall"
	elif learn and not recall and runtime >= 25:
		assert schedule == "learn - consolidate"
	elif learn and not recall and runtime < 25:
		assert schedule == "learn"
	elif not learn and recall and runtime >= 25:
		assert schedule == "consolidate - recall"
	elif not learn and recall and runtime < 25:
		assert schedule == "recall"
	elif not learn and not recall and runtime >= 25:
		assert schedule == "consolidate"
	elif not learn and not recall and runtime < 25:
		assert schedule == "background"

	# see if values loaded for recall phase are valid
	if schedule == "learn - consolidate - recall":
		assert recipe.h[1][0] == pytest.approx(recipe.h_0.value_as(U.mV), 0.05) # early-phase weight has decayed at synapse 0->1
		assert recipe.z[1][0] > 0.4  # late-phase weight has built up at synapse 0->1 (NOTE: no `h_0` normalization here!)
		assert recipe.p[1] > epsilon # protein has built up at neuron 1

	# get spikes & test the number of spikes (unless the schedule is "consolidate" or "background", where no or few spikes occur)
	if not schedule == "consolidate" and not schedule == "background":
		spike_data = np.loadtxt(getDataPath(s_desc, "spikes.txt")).transpose()
		spikes_0 = spike_data[0][spike_data[1] == 0] # spike times of neuron 0 (exc.)
		spikes_1 = spike_data[0][spike_data[1] == 1] # spike times of neuron 1 (exc.)
		spikes_2 = spike_data[0][spike_data[1] == 2] # spike times of neuron 2 (exc.)
		spikes_3 = spike_data[0][spike_data[1] == 3] # spike times of neuron 3 (exc.)
		spikes_4 = spike_data[0][spike_data[1] == 4] # spike times of neuron 4 (inh.)
		print(f"len(spikes_0) = {len(spikes_0)}")
		print(f"len(spikes_1) = {len(spikes_1)}")
		print(f"len(spikes_2) = {len(spikes_2)}")
		print(f"len(spikes_3) = {len(spikes_3)}")
		print(f"len(spikes_4) = {len(spikes_4)}")
		if "learn" in schedule: # with learning protocol: many spikes expected
			assert len(spikes_0) > 130
			assert len(spikes_1) > 130
			assert len(spikes_2) > 130
			assert len(spikes_3) > 130
			assert len(spikes_4) > 40
		elif "recall" in schedule: # with recall protocol: relatively many spikes expected
			assert len(spikes_0) > 40
			assert len(spikes_1) > 40
			assert len(spikes_2) > 40
			assert len(spikes_3) > 40
			assert len(spikes_4) > 10
		else: # only background: few spikes expected
			assert len(spikes_0) + len(spikes_1) + len(spikes_2) + len(spikes_3) + len(spikes_4) > 0

###############################################################################	
# Test a basic protocol for early-phase plasticity in a small pre-defined network ('smallnet2')
@pytest.mark.parametrize("prot, config_file", [("learn_protocol", "config_smallnet2_basic_early.json"), \
                                               ("learn_protocol", "config_smallnet2_basic_early_det.json"), \
                                               ("recall_protocol", "test_config_smallnet2_basic_early_recall.json")])
def test_smallnet2_basic_early(prot, config_file):

	# configuration (with explicit input through learning and recall protocols)
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config["simulation"]["short_description"] # short description of the simulation
	dt = config["simulation"]["dt"]
	stim_prot = anc.completeProt(config["simulation"][prot])
	n_stim = len(stim_prot["explicit_input"]["stim_times"]) # the number of stimulus pulses
	config["simulation"]["sample_curr"] = 1 # retrieve Ornstein-Uhlenbeck stimulation current
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation 
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ('{config_file}')", platform = "CPU")

	# test the plasticity induced by some spikes evoked at determined times
	tb_before_last_spike = int((stim_prot["explicit_input"]["stim_times"][n_stim-1] + config["synapses"]["t_ax_delay"])/dt) # timestep right before the onset of the PSP caused by the last stimulus
	tb_before_last_Ca_increase = int((stim_prot["explicit_input"]["stim_times"][n_stim-1] + config["synapses"]["syn_exc_calcium_plasticity"]["t_Ca_delay"])/dt) # timestep right before the presynapse-induced calcium increase caused by the last stimulus
	data_stacked = np.loadtxt(getDataPath(s_desc, "traces.txt"))

	assert data_stacked[tb_before_last_spike+2][1] > config["neuron"]["V_rev"] + epsilon # upon PSP onset: membrane potential greater than V_rev
	assert data_stacked[tb_before_last_spike+2][2] == pytest.approx(0) # upon PSP onset: stimulation current equal zero (again)
	assert data_stacked[tb_before_last_Ca_increase+2][3] > recipe.h_0.value_as(U.mV) + epsilon # early-phase weight greater than h_0
	assert data_stacked[tb_before_last_Ca_increase+2][4] == pytest.approx(0) # late-phase weight equal to zero
	assert data_stacked[tb_before_last_Ca_increase+2][5] + epsilon < config["synapses"]["syn_exc_calcium_plasticity"]["theta_p"] # calcium amount less than theta_p
	assert data_stacked[tb_before_last_Ca_increase+2][5] > config["synapses"]["syn_exc_calcium_plasticity"]["theta_d"] + epsilon # calcium amount greater than theta_d
	assert data_stacked[tb_before_last_Ca_increase+2][6] > epsilon # signal triggering protein synthesis greater than zero
	assert data_stacked[tb_before_last_Ca_increase+2][7] == pytest.approx(0) # protein concentration equal to zero

###############################################################################	
# Test a basic protocol for late-phase plasticity in a small pre-defined network ('smallnet2')
@pytest.mark.parametrize("config_file, platform", [("config_smallnet2_basic_late.json", "CPU"),
                                                   #("config_smallnet2_basic_late.json", "GPU"),  # takes long
                                                   ("test_config_smallnet2_basic_late_noff.json", "CPU")])
#@pytest.mark.parametrize("config_file, platform", [("test_config_smallnet2_basic_late_noff.json", "GPU")])  # takes very long
def test_smallnet2_basic_late(config_file, platform):

	# configuration (with learning protocol)
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config["simulation"]["short_description"] + f", {platform}" # short description of the simulation
	dt = config["simulation"]["dt"]
	op = config["simulation"]["output_period"] # should be greater than 1 because of long learning stimulation phase
	stim_prot = anc.completeProt(config["simulation"]["learn_protocol"])
	config["simulation"]["short_description"] = s_desc
	config["simulation"]["sample_curr"] = 1 # retrieve Ornstein-Uhlenbeck stimulation current
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation 
	recipe = anc.arborNetworkConsolidation(config, add_info=f"{cyanText(func_name)} ('{config_file}')", platform = platform)

	# test the plasticity induced by strong stimulation
	data_stacked = np.loadtxt(getDataPath(s_desc, "traces.txt"))

	tb_before_stim_end = int((stim_prot["time_start"] + stim_prot["duration"])*1000/dt/op) # timestep right before the end of the stimulation

	tau_h = config["synapses"]["syn_exc_calcium_plasticity"]["tau_h"]
	theta_pro = config["synapses"]["syn_exc_calcium_plasticity"]["theta_pro"]
	p_max = config["synapses"]["syn_exc_calcium_plasticity"]["p_max"]
	h_stim_end = data_stacked[tb_before_stim_end][3] # early-phase weight upon the end of the stimulation
	tb_before_ps_end = tb_before_stim_end + int(tau_h / 0.1 * np.log((h_stim_end - recipe.h_0.value_as(U.mV)) / theta_pro)/dt/op) # timestep right before the protein synthesis ends (in case there are no fast-forward timesteps)

	assert data_stacked[tb_before_stim_end][5] > config["synapses"]["syn_exc_calcium_plasticity"]["theta_p"] + epsilon # calcium amount greater than theta_p -- at the end of the stimulation
	assert h_stim_end == pytest.approx(2*recipe.h_0.value_as(U.mV), 0.1) # early-phase weight built up fully -- at the end of the stimulation
	assert data_stacked[-1][4] == pytest.approx(recipe.h_0.value_as(U.mV), 0.1) # late-phase weight built up fully -- at the end of the whole simulation

	# no consolidation (schedule: learn+recall)
	if "_noff" in config_file:
		assert data_stacked[tb_before_ps_end][7] == pytest.approx(p_max, 0.15) # protein concentration equal to `p_max` -- at the end of the protein synthesis
		assert data_stacked[tb_before_ps_end+1][6] < theta_pro # signal triggering protein synthesis below threshold -- at the end of the protein synthesis
	# consolidation (schedule: learn - consolidate)
	else:
		assert data_stacked[tb_before_stim_end][7] == pytest.approx(p_max, 0.15) # protein concentration equal to `p_max` -- at the end of the stimulation
		assert data_stacked[tb_before_stim_end+1][6] == pytest.approx(recipe.h_0.value_as(U.mV), 0.1) # signal triggering protein synthesis at maximum -- at the end of the stimulation

###############################################################################
# Fixture: provides static variables to store the numbers of spikes that were found by test_def_max_activity() across tests
@pytest.fixture(scope='session')
def num_spikes_max_activity():
    spikes = {'conn': None, 'no_conn': None} # intitialize
    return spikes

