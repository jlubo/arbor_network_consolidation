#!/bin/python3

# Arbor simulation of memory consolidation in recurrent spiking neural networks, consisting of leaky integrate-and-fire neurons that are
# connected via current-based, plastic synapses (early phase based on calcium dynamics and late phase based on synaptic tagging and capture).
# Intended to reproduce the results of Luboeinski and Tetzlaff, Commun. Biol., 2021 (https://doi.org/10.1038/s42003-021-01778-y).

# Copyright 2021-2023 Jannik Luboeinski
# License: Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Contact: mail[at]jlubo.net

import arbor
from arbor import units as U
import numpy as np
import gc
import json
import argparse
import os
import subprocess
import traceback
import copy
from outputUtilities import getTimestamp, getFormattedTime, getTimeDiff, \
                            setDataPathPrefix, getDataPath, \
                            openLog, writeLog, writeAddLog, flushLog, closeLog, \
                            getSynapseId

setDataPathPrefix("data_")

###############################################################################
# completeProt
# Takes a, possibly incomplete, stimulation protocol dictionary and returns a complete
# protocol after adding keys that are missing. This allows to leave out unnecessary keys
# when defining protocols in the JSON config file.
# - prot: stimulation protocol, possibly incomplete
# - return: complete stimulation protocol
def completeProt(prot):
	
	if prot.get("scheme") is None:
		prot["scheme"] = ""

	if prot.get("time_start") is None:
		prot["time_start"] = 0

	if prot.get("duration") is None:
		prot["duration"] = 0

	if prot.get("freq") is None:
		prot["freq"] = 0

	if prot.get("N_stim") is None:
		prot["N_stim"] = 0

	if prot.get("I_0") is None:
		prot["I_0"] = 0

	if prot.get("sigma_WN") is None:
		prot["sigma_WN"] = 0

	if prot.get("explicit_input") is None \
	  or prot["explicit_input"].get("receivers") is None \
	  or prot["explicit_input"].get("stim_times") is None:
		prot["explicit_input"] = { "receivers": [ ], "stim_times": [ ] }

	return prot

###############################################################################
# protFormatted
# Takes a complete stimulation protocol dictionary and creates a formatted string for
# human-readable output.
# - prot: stimulation protocol
# - return: formatted string with information about the protocol
def protFormatted(prot):
	if not prot['scheme']:
		fmstr = "none"
	elif prot['scheme'] == "EXPLICIT":
		fmstr = f"{prot['scheme']}, pulses at {prot['explicit_input']['stim_times']} ms, to neurons {prot['explicit_input']['receivers']}"
	elif prot['scheme'] == "STET":
		fmstr = f"{prot['scheme']} (Poisson) to neurons {prot['explicit_input']['receivers']}, starting at {prot['time_start']} s"
	elif prot['I_0'] != 0:
		fmstr = f"{prot['scheme']} (OU), I_0 = {prot['I_0']} nA, sigma_WN = {prot['sigma_WN']} nA s^1/2"
	elif prot['duration'] != 0:
		fmstr = f"{prot['scheme']} (OU), {prot['N_stim']} input neurons at {prot['freq']} Hz, starting at {prot['time_start']} s, lasting for {prot['duration']} s"
	else:
		fmstr = f"{prot['scheme']} (OU), {prot['N_stim']} input neurons at {prot['freq']} Hz, starting at {prot['time_start']} s"
	
	return fmstr

###############################################################################
# getInputProtocol
# Defines specific input/stimulation protocols. Returns start and end time. If 'label' is provided,
# also returns event generator instances (may either implement an explicit schedule, or a protocol for 
# an 'ou_input' mechanism with the following behavior: if the value of the 'weight' parameter is 1, 
# is switched on, else if it is -1, stimulation is switched off).
# - protocol: protocol that is being used for stimulation
# - runtime: full runtime of the simulation (with unit)
# - dt : duration of one timestep (with unit)
# - explicit_input_times: list of times (in ms) at which explicit input pulses shall be given
# - label [optional]: label of the mechanism that receives the the stimulation
# - return: a list of 'arbor.event_generator' instances, the start time of the stimulus, and the end time of the stimulus in ms
def getInputProtocol(protocol, runtime, dt, explicit_input_times, label = ""):

	prot_name = protocol['scheme'] # name of the protocol (defining its structure)
	start_time = protocol['time_start']*U.s # time at which the stimulus starts

	if prot_name == "RECT":
		duration = protocol['duration']*U.s # duration of the stimulus
		end_time = start_time + duration + dt # time at which the stimulus ends

		if label:
			# create regular schedules to implement a stimulation pulse that lasts for 'duration'
			stim_on = arbor.event_generator(
			            label,
			            1,
			            arbor.regular_schedule(start_time, dt, start_time + dt))
			stim_off = arbor.event_generator(
			            label,
			            -1,
			            arbor.regular_schedule(start_time + duration, dt, start_time + duration + dt))
			
			return [stim_on, stim_off], start_time, end_time

		return [], start_time, end_time

	elif prot_name == "ONEPULSE":
		end_time = start_time + 100*U.ms + dt # time at which the stimulus ends

		if label:
			# create regular schedules to implement a stimulation pulse that lasts for 0.1 s
			stim_on = arbor.event_generator(
			            label,
			            1,
			            arbor.regular_schedule(start_time, dt, start_time + dt))
			stim_off = arbor.event_generator(
			            label,
			            -1,
			            arbor.regular_schedule(start_time + 100*U.ms, dt, end_time))
			
			return [stim_on, stim_off], start_time, end_time

		return [], start_time, end_time
		
	elif prot_name == "TRIPLET":
		end_time = start_time + 1100*U.ms + dt # time at which the stimulus ends in ms
		
		if label:
			# create regular schedules to implement pulses that last for 0.1 s each
			stim1_on = arbor.event_generator(
			             label,
			             1,
			             arbor.regular_schedule(start_time, dt, start_time + dt))
			stim1_off = arbor.event_generator(
			             label,
			             -1,
			             arbor.regular_schedule(start_time + 100*U.ms, dt, start_time + 100*U.ms + dt))
			stim2_on = arbor.event_generator(
			             label,
			             1,
			             arbor.regular_schedule(start_time + 500*U.ms, dt, start_time + 500*U.ms + dt))
			stim2_off = arbor.event_generator(
			             label,
			             -1,
			             arbor.regular_schedule(start_time + 600*U.ms, dt, start_time + 600*U.ms + dt))
			stim3_on = arbor.event_generator(
			             label,
			             1,
			             arbor.regular_schedule(start_time + 1000*U.ms, dt, start_time + 1000*U.ms + dt))
			stim3_off = arbor.event_generator(
			             label,
			             -1,
			             arbor.regular_schedule(start_time + 1100*U.ms, dt, end_time))
				  
			return [stim1_on, stim1_off, stim2_on, stim2_off, stim3_on, stim3_off], start_time, end_time
		
		return [], start_time, end_time

	elif prot_name == "STET":
		duration = protocol['duration']*U.s # duration of the stimulus (for "standard" STET: 1200 ms)
		last_pulse_start_time = start_time + duration

		t_start = np.linspace(start_time, last_pulse_start_time, num=3, endpoint=True) # start times of the pulses (in ms)
		t_end = np.linspace(start_time + 1000*U.ms, last_pulse_start_time + 1000*U.ms, num=3, endpoint=True) # end times of the pulses (in ms)

		end_time = t_end[-1]

		freq = protocol['freq']*U.Hz # average spike frequency in Hz
		seed = int(datetime.now().timestamp() * 1e6)

		# average number of spikes (random number drawn for every timestep, then filtered with probability):
		stimulus_times_exc = np.array([])
		rng = np.random.default_rng(seed)
		num_timesteps = np.int_(np.round_((t_end[0]-t_start[0])/dt))
		for i in range(len(t_start)):
			spike_mask = rng.random(size=num_timesteps) < freq*dt/1000.
			timestep_values = np.linspace(t_start[i], t_end[i], num=num_timesteps, endpoint=False)
			spikes = timestep_values[spike_mask]
			stimulus_times_exc = np.concatenate([stimulus_times_exc, spikes])

		if label:
			stim_explicit = arbor.event_generator(
					label,
					0.,
					arbor.explicit_schedule(stimulus_times_exc))

			return [stim_explicit], start_time, end_time
					
		return [], start_time, end_time
		
	elif prot_name == "FULL":
		end_time = runtime # time at which the stimulus ends

		if label:
			# create a regular schedule that lasts for the full runtime
			stim_on = arbor.event_generator(
			           label,
			           1,
			           arbor.regular_schedule(start_time, dt, start_time + dt))
				  
			return [stim_on], start_time, end_time

		return [], start_time, end_time

	elif prot_name == "EXPLICIT" and explicit_input_times.any():
		# depends on input pulses explicitly defined by 'explicit_input_times'
		start_time = np.min(explicit_input_times)*U.ms # time at which the stimulation starts
		end_time = np.max(explicit_input_times)*U.ms # time at which the stimulation ends

		if label:
			# create a schedule with explicitly defined pulses
			stim_explicit = arbor.event_generator(
			                 label,
			                 0.,
			                 arbor.explicit_schedule(explicit_input_times*U.ms))

			return [stim_explicit], start_time, end_time
		
		return [], start_time, end_time

	return [], np.nan*U.ms, np.nan*U.ms

#####################################
# NetworkRecipe
# Implementation of Arbor simulation recipe
class NetworkRecipe(arbor.recipe):

	# constructor
	# Sets up the recipe object; sets parameters from config dictionary (deep-copies the non-immutable structures to enable their modification)
	# - config: dictionary containing configuration data
	# - adap_dt: duration of one adapted timestep
	# - plasticity_mechanism: the plasticity mechanism to be used
	# - store_latest_state: specifies if latest synaptic state shall be probed and stored
	def __init__(self, config, adap_dt, plasticity_mechanism, store_latest_state):

		# The base C++ class constructor must be called first, to ensure that
		# all memory in the C++ class is initialized correctly. (see https://github.com/tetzlab/FIPPA/blob/main/STDP/arbor_lif_stdp.py)
		arbor.recipe.__init__(self)
		
		self.s_desc = config['simulation']['short_description'] # short description of the simulation
		self.N_exc = int(config["populations"]["N_exc"]) # number of neurons in the excitatory population
		self.N_inh = int(config["populations"]["N_inh"]) # number of neurons in the inhibitory population
		self.N_CA = int(config["populations"]["N_CA"]) # number of neurons in the cell assembly
		self.N_tot = self.N_exc + self.N_inh # total number of neurons (excitatory and inhibitory)
		self.p_c = config["populations"]["p_c"] # probability of connection
		self.p_r = config["populations"]["p_r"] # probability of recall stimulation
		
		self.props = arbor.neuron_cable_properties() # initialize the cell properties to match Neuron's defaults 
		                                             # (cf. https://docs.arbor-sim.org/en/v0.5.2/tutorial/single_cell_recipe.html)
		
		cat = arbor.load_catalogue("./custom-catalogue.so") # load the catalogue of custom mechanisms
		cat.extend(arbor.default_catalogue(), "") # add the default catalogue
		cat.extend(arbor.stochastic_catalogue(), "") # add the stochastic catalogue
		self.props.catalogue = cat

		self.plasticity_mechanism = plasticity_mechanism #+ "/sps_=sps" # set the plasticity mechanism and the particle for diffusion

		self.runtime = config["simulation"]["runtime"]*U.s # runtime of the simulation in biological time
		self.std_dt = config["simulation"]["dt"]*U.ms # duration of one standard timestep (for rich computation)
		self.adap_dt = adap_dt # duration of one adapted timestep
		self.neuron_config = config["neuron"]
		self.syn_config = config["synapses"]
		self.syn_plasticity_config = config["synapses"]["syn_exc_calcium_plasticity"]
		self.h_0 = self.syn_plasticity_config["h_0"]*U.mV
		self.D = 1*U.m2/U.s # diffusivity
		self.w_ei = config["populations"]["w_ei"]*U.mV
		self.w_ie = config["populations"]["w_ie"]*U.mV
		self.w_ii = config["populations"]["w_ii"]*U.mV
		self.learn_prot = completeProt(config["simulation"]["learn_protocol"]) # protocol for learning stimulation as a dictionary with the keys "time_start" (starting time), "scheme" (scheme of pulses), and "freq" (stimulation frequency)
		self.recall_prot = completeProt(config["simulation"]["recall_protocol"]) # protocol for recall stimulation as a dictionary with the keys "time_start" (starting time), "scheme" (scheme of pulses), and "freq" (stimulation frequency)
		self.bg_prot = completeProt(config["simulation"]["bg_protocol"]) # protocol for background input as a dictionary with the keys "time_start" (starting time), "scheme" (scheme of pulses), "I_0" (mean), and "sigma_WN" (standard deviation)
		self.sample_gid_list = config['simulation']['sample_gid_list'] # list of the neurons that are to be probed (given by number/gid)
		self.sample_syn_list = config['simulation']['sample_syn_list'] # list of the synapses that are to be probed (one synapse per neuron only, given by its internal number respective to a neuron in sample_gid); -1: none
		
		if config['populations']['conn_file']: # if a connections file is specified, load the connectivity matrix from that file
			self.conn_matrix = np.loadtxt(config['populations']['conn_file']).T
		else: # there is no pre-defined connectivity matrix -> generate one
			rng = np.random.default_rng() # random number generator
			self.conn_matrix = rng.random((self.N_tot, self.N_tot)) <= self.p_c # two-dim. array of booleans indicating the existence of any incoming connection
			self.conn_matrix[np.identity(self.N_tot, dtype=bool)] = 0 # remove self-couplings
		np.savetxt(getDataPath(self.s_desc, "connections.txt"), self.conn_matrix.T, fmt="%.0f") # save the connectivity matrix

		## set diffusing particles 
		self.props.set_ion("sps_", valence=1, int_con=0*U.mM, ext_con=0*U.mM, rev_pot=0*U.mV) #diff=self.D) # signal triggering protein synthesis
		self.props.set_ion("pc_", valence=1, int_con=0*U.mM, ext_con=0*U.mM, rev_pot=0*U.mV) # common pool of plasticity-related proteins to diffuse
		# name, charge, internal_concentration, external_concentration, reversal_potential, reversal_potential_method, diffusivity

		# for testing
		self.exc_neurons_counter = 0
		self.inh_neurons_counter = 0
		self.connections_counter = 0

		# states to store/load
		self.h = None
		self.z = None
		self.p = None
		self.max_num_trace_probes = 0
		self.max_num_latest_state_probes = None
		self.store_latest_state = store_latest_state

	# loadState
	# Loads the relevant variables of the latest state of all neurons and synapses (from the end of the previous simulation phase)
	# - prev_phase: number specifying the previous phase from which to load the data
	def loadState(self, prev_phase):

		state_path = getDataPath(self.s_desc, f"state_after_phase_{prev_phase}")

		# loading the adjacency matrices for h, z, and p
		self.h = np.loadtxt(os.path.join(state_path, "h_synapses.txt")).T
		self.z = np.loadtxt(os.path.join(state_path, "z_synapses.txt")).T
		self.p = np.loadtxt(os.path.join(state_path, "p_compartments.txt"))

	# cell_kind
	# Defines the kind of the neuron given by gid
	# - gid: global identifier of the cell
	# - return: type of the cell
	def cell_kind(self, gid):
		
		return arbor.cell_kind.cable # note: implementation of arbor.cell_kind.lif is not ready to use yet

	# cell_description
	# Defines the morphology, cell mechanism, etc. of the neuron given by gid
	# - gid: global identifier of the cell
	# - return: description of the cell
	def cell_description(self, gid):

		# cylinder morphology
		tree = arbor.segment_tree()
		radius_µm = self.neuron_config["radius"] # radius of cylinder in µm
		height_µm = 2*radius_µm # height of cylinder in µm
		tree.append(arbor.mnpos,
		            arbor.mpoint(-1/2 * height_µm, 0, 0, radius_µm),
		            arbor.mpoint(1/2 * height_µm, 0, 0, radius_µm),
		            tag=1)
		labels = arbor.label_dict({"center": "(location 0 0.5)"})
		area = 2 * np.pi * radius_µm*U.um * height_µm*U.um # surface area of the cylinder (excluding the circle-shaped ends, since Arbor does not consider current flux there)
		area_cm2 = 2 * np.pi * radius_µm * height_µm * (1e-4)**2  # surface area of the cylinder in cm^2
		volume = np.pi * radius_µm**2*U.um2 * height_µm*U.um # volume of the cylinder
		self.volume = volume
		#i_factor = (1/area) * U.nA/(U.mA/U.cm2) # current to current density conversion factor (nA to mA/cm^2; necessary for point neurons)
		i_factor = (1e-9/1e-3) / area_cm2 # current to current density conversion factor (nA to mA/cm^2; necessary for point neurons)
		C_mem = self.neuron_config["C_mem"]*U.F # membrane capacitance
		c_mem = C_mem / area # specific membrane capacitance in F/m^2, computed from absolute capacitance of a point neuron

		# initialize decor
		decor = arbor.decor()
		decor.discretization(arbor.cv_policy("(max-extent 100)"))

		# neuronal dynamics
		decor.set_property(Vm=self.neuron_config["V_init"]*U.mV, cm=c_mem)
		mech_neuron = arbor.mechanism(self.neuron_config["mechanism"])
		R_leak = self.neuron_config["R_leak"]*U.MOhm
		R_reset = self.neuron_config["R_reset"]*U.MOhm
		tau_mem = R_leak * C_mem # membrane time constant
		V_rev = self.neuron_config["V_rev"]*U.mV
		V_reset = self.neuron_config["V_reset"]*U.mV
		V_th = self.neuron_config["V_th"]*U.mV
		t_ref = self.neuron_config["t_ref"]*U.ms
		U_substance_amount = U.nano*U.nano*U.mol #1e-18*U.mol
		theta_pro = self.syn_plasticity_config["theta_pro"]*U_substance_amount
		p_max = self.syn_plasticity_config["p_max"]*U.mM
		mech_neuron.set("R_leak", R_leak.value_as(U.MOhm))
		mech_neuron.set("R_reset", R_reset.value_as(U.MOhm))
		mech_neuron.set("I_0", 0) # set to zero (background input is applied via OU process ou_bg)
		mech_neuron.set("i_factor", i_factor)
		mech_neuron.set("V_rev", V_rev.value_as(U.mV))
		mech_neuron.set("V_reset", V_reset.value_as(U.mV))
		mech_neuron.set("V_th", V_th.value_as(U.mV))
		mech_neuron.set("t_ref", t_ref.value_as(U.ms))
		mech_neuron.set("theta_pro", theta_pro.value_as(U_substance_amount))
		mech_neuron.set("p_max", p_max.value_as(U.mM))

		# diffusion of particles/signals
		decor.set_ion("sps_", int_con=0*U.mM, diff=self.D) # signal to trigger protein synthesis
		decor.set_ion("pc_", int_con=0*U.mM, diff=self.D) # proteins

		# excitatory neurons
		if gid < self.N_exc:
			# set neuronal state variables to values loaded from previous state of the network
			if self.p is not None:
				mech_neuron.set('pc_init', self.p[gid])

			# parameter output
			if gid == 0:
				writeAddLog("area =", area, "µm^2")
				writeAddLog("volume =", volume, "µm^3")
				writeAddLog("i_factor =", i_factor, "(mA/cm^2)/nA")
				writeAddLog("c_mem =", c_mem, "F/m^2")
				writeAddLog("tau_mem =", tau_mem, "ms")
		
			# create plastic excitatory exponential synapse
			mech_expsyn_exc = arbor.mechanism(self.plasticity_mechanism)
			
			# set standard parameters
			mech_expsyn_exc.set('h_0', self.h_0.value_as(U.mV))
			mech_expsyn_exc.set('theta_tag', self.syn_plasticity_config["theta_tag"])
			#mech_expsyn_exc.set('area', area.value_as(U.um**2))

			if not self.plasticity_mechanism[-3:] == "_ff":
				mech_expsyn_exc.set('R_mem', R_leak.value_as(U.MOhm))
				mech_expsyn_exc.set('tau_syn', self.syn_config["tau_syn"])
				mech_expsyn_exc.set('Ca_pre', self.syn_plasticity_config["Ca_pre"])
				mech_expsyn_exc.set('Ca_post', self.syn_plasticity_config["Ca_post"])
				mech_expsyn_exc.set('theta_p', self.syn_plasticity_config["theta_p"])
				mech_expsyn_exc.set('theta_d', self.syn_plasticity_config["theta_d"])
				mech_expsyn_exc.set('sigma_pl', self.syn_plasticity_config["sigma_pl"])

			exc_connections = np.array(self.conn_matrix[gid][0:self.N_exc], dtype=bool) # array of booleans indicating all incoming excitatory connections	
			exc_pre_neurons = np.arange(self.N_exc)[exc_connections] # array of excitatory presynaptic neurons indicated by their gid
			#inc_exc_connections = np.sum(self.conn_matrix[gid][0:self.N_exc], dtype=int) # number of incoming excitatory connections
			inc_exc_connections = len(exc_pre_neurons)

			# place synapses; set synaptic state variables to values loaded from previous state of the network
			for n in reversed(range(inc_exc_connections)): ### NOTE THE REVERSED ORDER (SEE BELOW) !!!
				if (self.h is not None and self.z is not None):
					mech_expsyn_exc.set('h_init', self.h[gid][exc_pre_neurons[n]])
					mech_expsyn_exc.set('z_init', self.z[gid][exc_pre_neurons[n]])
					if self.N_exc <= 4: # only for small networks
						writeAddLog(f"Setting loaded values for synapse {exc_pre_neurons[n]}->{gid} (neuron {gid}, inc. " +
						            f"synapse {n})..." +
							        f"\n  h = {self.h[gid][exc_pre_neurons[n]]}" +
							        f"\n  z = {self.z[gid][exc_pre_neurons[n]]}")
				# add synapse
				decor.place('"center"', arbor.synapse(mech_expsyn_exc), "syn_ee_calcium_plasticity") # place synapse at the center of the soma (because: point neuron)
			#writeLog("Placed", inc_exc_connections, "incoming E->E synapses for neuron", gid)

			# non-plastic inhibitory exponential synapse
			mech_expsyn_inh = arbor.mechanism('expsyn_curr')
			mech_expsyn_inh.set('w', -self.w_ie.value_as(U.mV) * self.h_0.value_as(U.mV))
			mech_expsyn_inh.set('R_mem', R_leak.value_as(U.MOhm))
			mech_expsyn_inh.set('tau', self.syn_config["tau_syn"])

			inc_inh_connections = np.sum(self.conn_matrix[gid][self.N_exc:self.N_tot], dtype=int) # number of incoming inhibitory connections
			for i in range(inc_inh_connections):
				decor.place('"center"', arbor.synapse(mech_expsyn_inh), "syn_ie") # place synapse at the center of the soma (because: point neuron)
			
			self.exc_neurons_counter += 1 # for testing
			
		# inhibitory neurons
		else:			
			# non-plastic excitatory exponential synapse
			mech_expsyn_exc = arbor.mechanism('expsyn_curr')
			mech_expsyn_exc.set('w', self.w_ei.value_as(U.mV) * self.h_0.value_as(U.mV))
			mech_expsyn_exc.set('R_mem', R_leak.value_as(U.MOhm))
			mech_expsyn_exc.set('tau', self.syn_config["tau_syn"])
			inc_exc_connections = np.sum(self.conn_matrix[gid][0:self.N_exc], dtype=int) # number of incoming excitatory connections
			for i in range(inc_exc_connections):
				decor.place('"center"', arbor.synapse(mech_expsyn_exc), "syn_ei") # place synapse at the center of the soma (because: point neuron)
			
			# non-plastic inhibitory exponential synapse
			mech_expsyn_inh = arbor.mechanism('expsyn_curr')
			mech_expsyn_inh.set('w', -self.w_ii.value_as(U.mV) * self.h_0.value_as(U.mV))
			mech_expsyn_inh.set('R_mem', R_leak.value_as(U.MOhm))
			mech_expsyn_inh.set('tau', self.syn_config["tau_syn"])

			inc_inh_connections = np.sum(self.conn_matrix[gid][self.N_exc:self.N_tot], dtype=int) # number of incoming inhibitory connections
			for i in range(inc_inh_connections):
				decor.place('"center"', arbor.synapse(mech_expsyn_inh), "syn_ii") # place synapse at the center of the soma (because: point neuron)
			
			self.inh_neurons_counter += 1 # for testing
			
		# learning stimulation to all neurons in the assembly core ('as' and 'ans' subpopulations); input current described by Ornstein-Uhlenbeck process
		#  accounting for a population of neurons
		if gid < self.N_CA:
			mech_ou_learn_stim = arbor.mechanism('ou_input')
			ou_mu = self.learn_prot['N_stim'] * self.learn_prot['freq']*U.Hz * 1*U.s * self.h_0 / R_leak
			#ou_sigma = np.sqrt(self.learn_prot['N_stim'] * self.learn_prot['freq']*U.Hz / (2 * self.syn_config['tau_syn']*U.ms)) * 1*U.s * self.h_0/R_leak
			ou_sigma = np.sqrt(1000.0 * self.learn_prot['N_stim'] * self.learn_prot['freq'] / (2 * self.syn_config['tau_syn'])) * self.h_0.value_as(U.mV)/R_leak.value_as(U.MOhm) * U.nA
			mech_ou_learn_stim.set('mu', ou_mu.value_as(U.nA))
			#mech_ou_learn_stim.set('sigma', ou_sigma.value_as(U.nA / np.sqrt(U.s))) # fractional units not supported!
			mech_ou_learn_stim.set('sigma', ou_sigma.value_as(U.nA)) # technically incorrect unit (see line above)
			mech_ou_learn_stim.set('tau', self.syn_config["tau_syn"])
			decor.place('"center"', arbor.synapse(mech_ou_learn_stim), "ou_learn_stim")

		# recall stimulation to some neurons in the assembly core ('as' subpopulation); input current described by Ornstein-Uhlenbeck process
		#  accounting for a population of neurons
		if gid < self.p_r*self.N_CA:
			mech_ou_recall_stim = arbor.mechanism('ou_input')
			ou_mu = self.recall_prot['N_stim'] * self.recall_prot['freq']*U.Hz * 1*U.s * self.h_0 / R_leak
			ou_sigma = np.sqrt(1000.0 * self.recall_prot['N_stim'] * self.recall_prot['freq'] / (2 * self.syn_config['tau_syn'])) * self.h_0.value_as(U.mV)/R_leak.value_as(U.MOhm) * U.nA
			mech_ou_recall_stim.set('mu', ou_mu.value_as(U.nA))
			#mech_ou_recall_stim.set('sigma', ou_sigma.value_as(U.nA / np.sqrt(U.s))) # fractional units not supported!
			mech_ou_recall_stim.set('sigma', ou_sigma.value_as(U.nA)) # technically incorrect unit (see line above)
			mech_ou_recall_stim.set('tau', self.syn_config["tau_syn"])
			decor.place('"center"', arbor.synapse(mech_ou_recall_stim), "ou_recall_stim")
			
		# background input current to all neurons; input current described by Ornstein-Uhlenbeck process
		#  with given mean and standard deviation
		mech_ou_bg = arbor.mechanism('ou_input')
		ou_mu = self.bg_prot["I_0"]*U.nA
		ou_sigma = self.bg_prot['sigma_WN'] * np.sqrt(1000. / (2 * self.syn_config['tau_syn'])) * U.nA
		mech_ou_bg.set('mu', ou_mu.value_as(U.nA))
		#mech_ou_bg.set('sigma', ou_sigma.value_as(U.nA / np.sqrt(U.s))) # fractional units not supported!
		mech_ou_bg.set('sigma', ou_sigma.value_as(U.nA)) # technically incorrect unit (see line above)
		mech_ou_bg.set('tau', self.syn_config["tau_syn"])
		decor.place('"center"', arbor.synapse(mech_ou_bg), "ou_bg")
		
		# additional excitatory delta synapse for explicit input
		mech_deltasyn_exc = arbor.mechanism('deltasyn')
		mech_deltasyn_exc.set('g_spike', 1000*(V_th.value_as(U.mV)-V_reset.value_as(U.mV))*np.exp(self.std_dt.value_as(U.ms)/tau_mem.value_as(U.ms))) # choose sufficently large increase in conductance
		decor.place('"center"', arbor.synapse(mech_deltasyn_exc), "explicit_input")
			
		# place spike detector
		decor.place('"center"', arbor.threshold_detector(V_th), "spike_detector")

		# paint neuron mechanism
		decor.paint('(all)', arbor.density(mech_neuron))
			
		return arbor.cable_cell(tree, decor, labels)
		
	# connections_on
	# Defines the list of incoming synaptic connections to the neuron given by gid
	# - gid: global identifier of the cell
	# - return: connections to the given neuron
	def connections_on(self, gid):
		connections_list = []

		rr = arbor.selection_policy.round_robin
		rr_halt = arbor.selection_policy.round_robin_halt
		
		connections = self.conn_matrix[gid]
		assert connections[gid] == 0 # check that there are no self-couplings

		self.connections_counter += np.sum(connections)
		
		exc_connections = np.array(connections*np.concatenate((np.ones(self.N_exc, dtype=np.int8), np.zeros(self.N_inh, dtype=np.int8)), axis=None), dtype=bool) # array of booleans indicating all incoming excitatory connections
		inh_connections = np.array(connections*np.concatenate((np.zeros(self.N_exc, dtype=np.int8), np.ones(self.N_inh, dtype=np.int8)), axis=None), dtype=bool) # array of booleans indicating all incoming inhibitory connections
				
		assert not np.any(np.logical_xor(np.logical_or(exc_connections, inh_connections), connections)) # test if 'exc_connections' and 'inh_connections' together yield 'connections' again
		
		exc_pre_neurons = np.arange(self.N_tot)[exc_connections] # array of excitatory presynaptic neurons indicated by their gid
		inh_pre_neurons = np.arange(self.N_tot)[inh_connections] # array of inhibitory presynaptic neurons indicated by their gid

		assert np.logical_and(np.all(exc_pre_neurons >= 0), np.all(exc_pre_neurons < self.N_exc)) # test if the excitatory neuron numbers are in the correct range
		assert np.logical_and(np.all(inh_pre_neurons >= self.N_exc), np.all(inh_pre_neurons < self.N_tot)) # test if the inhibitory neuron numbers are in the correct range
		
		# delay constants
		#d0 = self.syn_config["t_ax_delay"] # delay time of the postsynaptic potential in ms
		d0 = max(self.adap_dt.value_as(U.ms), self.syn_config["t_ax_delay"])*U.ms # delay time of the postsynaptic potential in ms
		#d1 = self.syn_plasticity_config["t_Ca_delay"] # delay time of the calcium increase in ms (only for plastic synapses)
		d1 = max(self.adap_dt.value_as(U.ms), self.syn_plasticity_config["t_Ca_delay"])*U.ms # delay time of the calcium increase in ms (only for plastic synapses)		
		
		# excitatory neurons
		if gid < self.N_exc:
								
			# incoming excitatory synapses
			for src in exc_pre_neurons:
				#connections_list.append(arbor.connection((src, "spike_detector"), ("syn_ee_calcium_plasticity", rr), 1, d0)) # for postsynaptic potentials
				connections_list.append(arbor.connection((src, "spike_detector"), ("syn_ee_calcium_plasticity", rr_halt), 1, d0)) # for postsynaptic potentials
				connections_list.append(arbor.connection((src, "spike_detector"), ("syn_ee_calcium_plasticity", rr), -1, d1)) # for plasticity-related calcium dynamics
			
			# incoming inhibitory synapses
			for src in inh_pre_neurons:
				connections_list.append(arbor.connection((src,"spike_detector"), ("syn_ie", rr), 1, d0))
				  
		# inhibitory neurons
		else:
			# incoming excitatory synapses
			for src in exc_pre_neurons:
				connections_list.append(arbor.connection((src,"spike_detector"), ("syn_ei", rr), 1, d0))
				
			# incoming inhibitory synapses
			for src in inh_pre_neurons:
				connections_list.append(arbor.connection((src,"spike_detector"), ("syn_ii", rr), 1, d0))
		
		if self.N_exc <= 4: # only for small networks
			writeAddLog(f"Setting connections for gid = {gid}...")
			for conn in connections_list:
				writeAddLog(f"  {conn}")
		return connections_list

	# event_generators
	# Event generators for input to synapses
	# - gid: global identifier of the cell
	# - return: events generated from Arbor schedule
	def event_generators(self, gid):
		inputs = []

		# background input current to all neurons
		stim, _, _ = getInputProtocol(self.bg_prot, self.runtime, self.std_dt,
		                              np.array(self.bg_prot["explicit_input"]["stim_times"]), "ou_bg")
		inputs.extend(stim)

		# explicitly specified pulses for learning stimulation to defined receiving neurons
		if (self.learn_prot["scheme"] == "EXPLICIT" and self.learn_prot["explicit_input"]["receivers"]) \
		  or self.learn_prot["scheme"] == "STET":
		
			if gid in self.learn_prot["explicit_input"]["receivers"]:

				stim, _, _ = getInputProtocol(self.learn_prot, self.runtime, self.std_dt,
				                              np.array(self.learn_prot["explicit_input"]["stim_times"]), "explicit_input")
				inputs.extend(stim)
		
		# Ornstein-Uhlenbeck learning stimulation to cell assembly neurons ('as' and 'ans' subpopulations)
		elif gid < self.N_CA:
			
			stim, _, _ = getInputProtocol(self.learn_prot, self.runtime, self.std_dt,
			                              np.array([]), "ou_learn_stim")
			inputs.extend(stim)

		# explicitly specified pulses for recall stimulation to defined receiving neurons
		if self.recall_prot["scheme"] == "EXPLICIT" and self.recall_prot["explicit_input"]["receivers"] \
		  or self.recall_prot["scheme"] == "STET":
		
			if gid in self.recall_prot["explicit_input"]["receivers"]:

				stim, _, _ = getInputProtocol(self.recall_prot, self.runtime, self.std_dt,
				                              np.array(self.recall_prot["explicit_input"]["stim_times"]), "explicit_input")
				inputs.extend(stim)
			
		# Ornstein-Uhlenbeck recall stimulation to cell assembly neurons ('as' subpopulation)
		elif gid < self.p_r*self.N_CA:
	
			stim, _, _ = getInputProtocol(self.recall_prot, self.runtime, self.std_dt,
			                              np.array([]), "ou_recall_stim")
			inputs.extend(stim)			
			
		return inputs
		
	# global_properties
	# Sets properties that will be applied to all neurons of the specified kind
	# - gid: global identifier of the cell
	# - return: the cell properties 
	def global_properties(self, kind): 

		assert kind == arbor.cell_kind.cable # assert that all neurons are technically cable cells

		return self.props
	
	# num_cells
	# - return: the total number of cells in the network
	def num_cells(self):
		
		return self.N_tot

	# set_max_num_trace_probes
	# Sets the maximal number of probes to measure the traces of a particular neuron or synapse
	def set_max_num_trace_probes(self, num):
		if num > self.max_num_trace_probes:
			self.max_num_trace_probes = num

	# get_max_num_trace_probes
	# - return: the number of probes to measure the traces of a particular neuron or synapse
	def get_max_num_trace_probes(self):
		return self.max_num_trace_probes

	# set_max_num_latest_state_probes
	# Sets the maximal number of probes to measure the latest synaptic state of a neuron
	def set_max_num_latest_state_probes(self, num):
		if self.max_num_latest_state_probes is None \
			or num > self.max_num_latest_state_probes:
			self.max_num_latest_state_probes = num

	# get_max_num_latest_state_probes
	# - return: the number of probes to measure the latest synaptic state of a neuron
	def get_max_num_latest_state_probes(self):
		if self.max_num_latest_state_probes is None:
			return 0
		return self.max_num_latest_state_probes

	# probes
	# Sets the probes to measure neuronal and synaptic state -- WARNING: the indexing here is (for CPU) reversed to that used by 'sim.sample((gid, index), sched)'
	# - gid: global identifier of the cell
	# - return: the probes on the given cell
	def probes(self, gid):

		if self.N_exc <= 4: # only for small networks
			writeAddLog(f"Setting probes for gid = {gid}...")

		# loop over all potential excitatory presynaptic neurons to set probes for latest state
		latest_state_probes = []
		if self.store_latest_state and gid < self.N_exc:
			latest_state_probes.extend([arbor.cable_probe_density_state('"center"', self.neuron_config["mechanism"], "pc", "tag_pc_latest")])
			num_sample_syn = 0
			for j in range(self.N_exc):
				if self.conn_matrix[gid][j]:
					latest_state_probes.extend([arbor.cable_probe_point_state(num_sample_syn, self.plasticity_mechanism, "h", f"tag_h_latest_{num_sample_syn}"),
												arbor.cable_probe_point_state(num_sample_syn, self.plasticity_mechanism, "z", f"tag_z_latest_{num_sample_syn}")])
												#arbor.cable_probe_point_state(num_sample_syn, self.plasticity_mechanism, "pc", f"tag_pc_{num_sample_syn}")])
					num_sample_syn += 1
			self.set_max_num_latest_state_probes(len(latest_state_probes))

		# loop over all synapses whose traces are to be probed (for each 'gid' in 'sample_gid_list')
		all_trace_probes = []
		for i in range(len(self.sample_gid_list)):
			trace_probes = []

			if self.sample_gid_list[i] == gid:
				# for every neuron to be probed (upon first specification): get membrane potential, total current, and external input currents
				if self.sample_gid_list[:i+1].count(gid) == 1:
					trace_probes.extend([arbor.cable_probe_membrane_voltage('"center"', "tag_v"),
										arbor.cable_probe_total_ion_current_cell("tag_I_tot"),
										arbor.cable_probe_point_state_cell("ou_input", "I_ou", "tag_I_ou")])
				
				# gets the synapse identifier for the corresponding neuron identifier in 'sample_gid_list'
				sample_syn = getSynapseId(self.sample_syn_list, i)

				# for synapses to be probed: additionally get synaptic calcium concentration, early- and late-phase weight, and concentration of PRPs
				if sample_syn >= 0:

					trace_probes.extend([arbor.cable_probe_point_state(sample_syn, self.plasticity_mechanism, "Ca", f"tag_Ca_{sample_syn}"),
								         arbor.cable_probe_point_state(sample_syn, self.plasticity_mechanism, "h", f"tag_h_{sample_syn}"),
								         arbor.cable_probe_point_state(sample_syn, self.plasticity_mechanism, "z", f"tag_z_{sample_syn}"),
					                     #arbor.cable_probe_ion_diff_concentration('"center"', "sps_", f"tag_sps__{sample_syn}"),
					                     #arbor.cable_probe_point_state(sample_syn, self.plasticity_mechanism, "sps", f"tag_sps_{sample_syn}"),
										 arbor.cable_probe_density_state('"center"', self.neuron_config["mechanism"], "spsV", f"tag_spsV_{sample_syn}"),
					                     #arbor.cable_probe_point_state(sample_syn, self.plasticity_mechanism, "pc", f"tag_pc_{sample_syn}"),
										 #arbor.cable_probe_ion_diff_concentration('"center"', "pc_", f"tag_pc__{sample_syn}"),
					                     arbor.cable_probe_density_state('"center"', self.neuron_config["mechanism"], "pc", f"tag_pc__{sample_syn}")])

				if self.N_exc <= 4: # only for small networks
					writeAddLog(f"  set probes for sample_syn = {sample_syn}")
	
			self.set_max_num_trace_probes(len(trace_probes))
			all_trace_probes.extend(trace_probes)

		if self.N_exc <= 4: # only for small networks
			writeAddLog("  max_num_trace_probes =", self.get_max_num_trace_probes())
			writeAddLog("  max_num_latest_state_probes =", self.get_max_num_latest_state_probes())

		#return [*all_trace_probes, *latest_state_probes] # see the warning above
		return [*latest_state_probes, *all_trace_probes]

#####################################
# runSimPhase
# Run a simulation phase given a recipe, runtime, and timestep.
# - phase: number of the current simulation phase (should equal to 1 upon first call)
# - recipe: the Arbor recipe to be used
# - config: configuration of model and simulation parameters (as a dictionary from JSON format)
# - rseed: random seed
# - t_final_red: final simulated time (reduced for the current phase)
# - adap_dt: the adapted timestep
# - platform: the hardware platform to be used (options: "CPU", "GPU")
# - plotting: specifies whether plots should be created or not
# - return: spike data and trace data
def runSimPhase(phase, recipe, config, rseed, t_final_red, adap_dt, platform, plotting):
	###############################################
	# initialize
	s_desc = config['simulation']['short_description'] # short description of the simulation, including hashtags for benchmarking
	sample_gid_list = config['simulation']['sample_gid_list'] # list of the neurons that are to be probed (given by number/gid)
	sample_syn_list = config['simulation']['sample_syn_list'] # list of the synapses that are to be probed (one synapse per neuron only, given by its internal number respective to a neuron in sample_gid); -1: none
	sample_curr = config['simulation']['sample_curr'] # pointer to current data (0: total membrane current, 1: OU stimulation current, 2: OU background noise current)
	output_period = config['simulation']['output_period'] # sampling size in timesteps (every "output_period-th" timestep, data will be recorded for plotting)
	loc = 0 # for single-compartment neurons, there is only one location

	###############################################
	# prepare domain decomposition and simulation
	writeLog("CPU:", subprocess.check_output("lscpu | grep \"^Model name:\"", shell=True).decode()
	                            .replace('\n', ' ')
	                            .replace('Model name:', '')
	                            .strip())
	if platform == "CPU":
		alloc = arbor.proc_allocation(threads=1, gpu_id=None) # select one thread and no GPU (default; cf. https://docs.arbor-sim.org/en/v0.7/python/hardware.html#arbor.proc_allocation)
		hint = arbor.partition_hint()
		hint.cpu_group_size = 1 #recipe.num_cells()
		context = arbor.context(alloc, mpi=None) # constructs a local context without MPI connection
		#context = arbor.context(threads="avail_threads", gpu_id=0)
		writeLog("GPU: none")
	elif platform == "GPU":
		#alloc = arbor.proc_allocation(threads=1, gpu_id=None) # select one thread and no GPU (default; cf. https://docs.arbor-sim.org/en/v0.7/python/hardware.html#arbor.proc_allocation)
		#alloc = arbor.proc_allocation(threads="avail_threads", gpu_id=0) # select one thread and no GPU (default; cf. https://docs.arbor-sim.org/en/v0.7/python/hardware.html#arbor.proc_allocation)
		hint = arbor.partition_hint()
		hint.prefer_gpu = True
		hint.gpu_group_size = 500 #recipe.num_cells()
		#context = arbor.context(alloc, mpi=None) # constructs a local context without MPI connection
		context = arbor.context(threads=1, gpu_id=0)
		writeLog("GPU:", subprocess.check_output("nvidia-smi -L", shell=True).decode()
		                           .replace('\n', '\n     ')
								   .strip())
	else:
		raise ValueError(f"Unknown platform: '{platform}'")
	meter_manager = arbor.meter_manager()
	meter_manager.start(context)
	hints = {arbor.cell_kind.cable: hint}
	domains = arbor.partition_load_balance(recipe, context, hints) # constructs a domain_decomposition that distributes the cells in the model described by an arbor.recipe over the distributed and local hardware resources described by an arbor.context (cf. https://docs.arbor-sim.org/en/v0.5.2/python/domdec.html#arbor.partition_load_balance)
	meter_manager.checkpoint('load-balance', context)
	sim = arbor.simulation(recipe, context, domains, seed = rseed)

	# output, to be printed only once
	if phase == 1: 
		num_conn_str = f"Number of connections: {int(recipe.connections_counter)}" + \
		               f" (expected value: {round(recipe.p_c * (recipe.N_tot**2 - recipe.N_tot), 1) if recipe.p_c > 0 else 'n/a'}"
		if config['populations']['conn_file']:
			num_conn_str += f", loaded from '{config['populations']['conn_file']}')"
		else:
			num_conn_str += ", generated)"
		writeLog(num_conn_str)
		writeLog(context)
		writeLog(hint)
		writeLog(domains)

	# set metering checkpoint
	meter_manager.checkpoint('simulation-init', context)

	# create sampling schedules
	writeLog(f"Creating schedule with adap_dt = {adap_dt.value_as(U.ms)} ms, output_period = {output_period}.")
	reg_sched = arbor.regular_schedule(0*U.ms, output_period*adap_dt, t_final_red) # create schedule for recording traces (of either rich or fast-forward dynamics)
	final_sched = arbor.explicit_schedule([t_final_red-adap_dt]) # create schedule for recording the final timestep
	#sampl_policy = arbor.sampling_policy.lax # use exact policy, just to be safe!?
		
	###############################################
	# set handles -- WARNING: the indexing here is (for CPU) reversed to that used by the probe setting mechanism !!!
	handle_mem, handle_tot_curr, handle_stim_curr, handle_Ca_specific, handle_h_specific, handle_z_specific, handle_sps_specific, handle_p_specific = [], [], [], [], [], [], [], []
	handle_h_syn_latest, handle_z_syn_latest, handle_p_comp_latest = [], [], []
	synapse_mapping = [[] for i in range(recipe.N_exc)] # list that contains a list of matrix indices indicating the presynaptic neurons for each (excitatory) neuron -> sparse representation
	max_num_latest_state_probes = recipe.get_max_num_latest_state_probes() # the number of latest-state probes before specific sampling probes begin (for excitatory neurons)
	writeAddLog(f"Setting handles with max_num_latest_state_probes = {max_num_latest_state_probes}.")

	# loop over all compartments and all potential synapses to set handles for latest state
	if max_num_latest_state_probes > 0: # only if latest state probes exist
		for i in range(recipe.N_exc):

			handle_p_comp_latest.append(sim.sample(i, "tag_pc_latest", final_sched)) # neuronal protein amount
			if recipe.N_exc <= 4: # only for small networks
				writeAddLog(f"Setting handles for latest state of neuron {i}..." +
				            f"\n  handle(p) = {handle_p_comp_latest[-1]} " + str(sim.probe_metadata(i, "tag_pc_latest")))

			for j in reversed(range(recipe.N_exc)): ### NOTE THE REVERSED ORDER !!!

				# check if synapse is supposed to exist
				if recipe.conn_matrix[i][j]:

					num_sample_syn = len(synapse_mapping[i]) # number of synapses that have so far been found for postsynaptic neuron 'i'

					handle_h_syn_latest.append(sim.sample(i, f"tag_h_latest_{num_sample_syn}", final_sched)) # early-phase weight
					handle_z_syn_latest.append(sim.sample(i, f"tag_z_latest_{num_sample_syn}", final_sched)) # late-phase weight
					if recipe.N_exc <= 4: # only for small networks
						writeAddLog(f"Setting handles for latest state of synapse {j}->{i} (neuron {i}, inc. " +
						            f"synapse {num_sample_syn})..." +
						            f"\n  handle(h) = {handle_h_syn_latest[-1]} " + str(sim.probe_metadata(i, f"tag_h_latest_{num_sample_syn}")) +
						            f"\n  handle(z) = {handle_z_syn_latest[-1]} " + str(sim.probe_metadata(i, f"tag_z_latest_{num_sample_syn}")))

					synapse_mapping[i].append(j)

	# loop over elements in 'sample_gid_list' (and thereby, 'sample_syn_list') to set handles for specific sampling
	for i in range(len(sample_gid_list)):

		# retrieve the current neuron index
		sample_gid = sample_gid_list[i]

		# retrieve the synapse index by the corresponding neuron identifier
		sample_syn = getSynapseId(sample_syn_list, i)
		writeAddLog(f"Setting handles for specific sample #{i} (neuron {sample_gid}, " +
		            f"synapse {sample_syn})...")

		# for all neurons: get membrane potential and current(s)
		handle_mem.append(sim.sample(sample_gid, "tag_v", reg_sched)) # membrane potential
		handle_tot_curr.append(sim.sample(sample_gid, "tag_I_tot", reg_sched)) # total current
		handle_stim_curr.append(sim.sample(sample_gid, "tag_I_ou", reg_sched)) # stimulus current
		writeAddLog(f"  handle(V) = {handle_mem[-1]} " + str(sim.probe_metadata(sample_gid, "tag_v")) +
		            f"\n  handle(I_tot) = {handle_tot_curr[-1]} " + str(sim.probe_metadata(sample_gid, "tag_I_tot")) +
	                f"\n  handle(I_stim) = {handle_stim_curr[-1]} " + str(sim.probe_metadata(sample_gid, "tag_I_ou")))

		# for excitatory neurons with synapses to be probed: get synapse data
		if sample_syn >= 0: # synapse probes exist only if this condition is true
			handle_Ca_specific.append(sim.sample(sample_gid, f'tag_Ca_{sample_syn}', reg_sched)) # calcium amount
			handle_h_specific.append(sim.sample(sample_gid, f'tag_h_{sample_syn}', reg_sched)) # early-phase weight
			handle_z_specific.append(sim.sample(sample_gid, f'tag_z_{sample_syn}', reg_sched)) # late-phase weight
			handle_sps_specific.append(sim.sample(sample_gid, f'tag_spsV_{sample_syn}', reg_sched)) # signal triggering protein synthesis
			handle_p_specific.append(sim.sample(sample_gid, f'tag_pc__{sample_syn}', reg_sched)) # neuronal protein amount
			writeAddLog(f"  handle(Ca) = {handle_Ca_specific[-1]} " + str(sim.probe_metadata(sample_gid, f"tag_Ca_{sample_syn}")) +
			            f"\n  handle(h) = {handle_h_specific[-1]} " + str(sim.probe_metadata(sample_gid, f"tag_h_{sample_syn}")) +
			            f"\n  handle(z) = {handle_z_specific[-1]} " + str(sim.probe_metadata(sample_gid, f"tag_z_{sample_syn}")) +
			            f"\n  handle(sps) = {handle_sps_specific[-1]} " + str(sim.probe_metadata(sample_gid, f"tag_spsV_{sample_syn}")) +
			            f"\n  handle(p) = {handle_p_specific[-1]} " + str(sim.probe_metadata(sample_gid, f"tag_pc__{sample_syn}")))

	writeAddLog("Number of set handles:" +
	            f"\n  len(handle_h_syn_latest) = {len(handle_h_syn_latest)}" +
	            f"\n  len(handle_z_syn_latest) = {len(handle_z_syn_latest)}" +
	            f"\n  len(handle_p_comp_latest) = {len(handle_p_comp_latest)}" +
	            f"\n  len(handle_mem) = {len(handle_mem)}" +
	            f"\n  len(handle_tot_curr) = {len(handle_tot_curr)}" +
	            f"\n  len(handle_stim_curr) = {len(handle_stim_curr)}" +
	            f"\n  len(handle_Ca_specific) = {len(handle_Ca_specific)}" +
	            f"\n  len(handle_h_specific) = {len(handle_h_specific)}" +
	            f"\n  len(handle_z_specific) = {len(handle_z_specific)}" +
	            f"\n  len(handle_sps_specific) = {len(handle_sps_specific)}" +
	            f"\n  len(handle_p_specific) = {len(handle_p_specific)}")

	meter_manager.checkpoint('handles-set', context)
	#arbor.profiler_initialize(context) # can be enabled if Arbor has been compiled with -DARB_WITH_PROFILING=ON

	###############################################
	# run the simulation
	if plotting: # show the progress banner only if plotting is enabled (not on cluster)
		sim.progress_banner()

	sim.record(arbor.spike_recording.all)
	#sim.set_binning_policy(arbor.binning.regular, adap_dt)

	sim.run(tfinal=t_final_red, dt=adap_dt) # recall
	meter_manager.checkpoint('simulation-run', context)

	# output of metering and profiler summaries
	writeAddLog("Metering summary:\n", arbor.meter_report(meter_manager, context))
	#writeAddLog("Profiler summary:\n", arbor.profiler_summary()) # can be enabled if Arbor has been compiled with -DARB_WITH_PROFILING=ON

	###############################################
	# get and store the latest state of all compartments and synapses (for the next simulation phase)
	if max_num_latest_state_probes > 0: # only if latest state probes exist
		h_synapses = np.zeros((recipe.N_exc, recipe.N_exc))
		z_synapses = np.zeros((recipe.N_exc, recipe.N_exc))
		p_compartments = np.zeros(recipe.N_exc)

		state_path = getDataPath(s_desc, f"state_after_phase_{phase}")

		if not os.path.isdir(state_path): # if this directory does not exist yet
			os.mkdir(state_path)

		# loop over all compartments and synapses and set handles to get latest state
		n = 0 # 1-dim of synapse (for sparse representation)
		for i in range(recipe.N_exc):
			
			p_samples = sim.samples(handle_p_comp_latest[i])
			
			if len(p_samples) > 0:
				tmp, _ = p_samples[loc]
				p_compartments[i] = tmp[:, 1].squeeze()
			else:
				raise Exception(f"No data for handle_p_comp_latest[{i}] = {handle_p_comp_latest[i]}")

			# loop over synapses
			for num_sample_syn in range(len(synapse_mapping[i])):
				j = synapse_mapping[i][num_sample_syn]
				if recipe.N_exc <= 4: # only for small networks
					writeAddLog(f"Getting latest state of synapse {j}->{i} (neuron {i}, incoming synapse {num_sample_syn})...")
				if recipe.conn_matrix[i][j]:
					h_samples = sim.samples(handle_h_syn_latest[n])
					z_samples = sim.samples(handle_z_syn_latest[n])

					if len(h_samples) > 0:
						tmp, _ = h_samples[loc]
						h_synapses[i][j] = tmp[:, 1].squeeze()
					else:
						raise Exception(f"No data for handle_h_syn_latest[{n}] = {handle_h_syn_latest[n]}")

					if len(z_samples) > 0:
						tmp, _ = z_samples[loc]
						z_synapses[i][j] = tmp[:, 1].squeeze()
					else:
						raise Exception(f"No data for handle_z_syn_latest[{n}] = {handle_z_syn_latest[n]}")

					n += 1
				else:
					raise Exception(f"Entry with i = {i}, j = {j} not found in connectivity matrix.")

		np.savetxt(os.path.join(state_path, "h_synapses.txt"), h_synapses.T, fmt="%.4f")
		np.savetxt(os.path.join(state_path, "z_synapses.txt"), z_synapses.T, fmt="%.4f")
		np.savetxt(os.path.join(state_path, "p_compartments.txt"), p_compartments, fmt="%.4f")

	###############################################
	# get data traces	
	sample_gid_list = config['simulation']['sample_gid_list'] # list of the neurons that are to be probed (given by number/gid)
	sample_curr = config['simulation']['sample_curr'] # pointer to current data (0: total membrane current, 1: OU stimulation current, 2: OU background noise current)

	data, times = [], []
	data_mem, data_curr, data_Ca, data_h = [], [], [], []
	data_z, data_sps, data_p = [], [], []
	dimin = 0

	# loop over elements in 'sample_gid_list' (and thereby, 'sample_syn_list') to get traces
	for i in range(len(sample_gid_list)):
		# retrieve the current neuron index
		sample_gid = sample_gid_list[i]

		# retrieve the synapse index by the corresponding neuron identifier
		sample_syn = getSynapseId(sample_syn_list, i)
		writeLog(f"Getting traces of sample #{i} (neuron {sample_gid}, synapse {sample_syn})...")

		# get neuronal data
		data, _ = sim.samples(handle_mem[i])[loc]
		if len(sim.samples(handle_mem[i])) <= 0: 
			raise ValueError(f"No data for handle_mem[{i}] (= {handle_mem[i]}).")
		times = data[:, 0]
		data_mem.append(data[:, 1])
		writeAddLog(f"  Read from handle {handle_mem[i]} (V, {bool(sim.samples(handle_mem[i]))})")
		if sample_curr > 0: # get OU stimulation current specified by sample_curr
			data_stim_curr, _ = sim.samples(handle_stim_curr[i])[loc]
			if len(data_stim_curr[0]) > sample_curr:
				data_curr.append(data_stim_curr[:, sample_curr])
			else:
				writeLog(f"  No data for handle_stim_curr[{i}] = {handle_stim_curr[i]} (sample_curr={sample_curr})")
				data_curr.append(np.zeros(len(data_stim_curr[:, 0])))
			writeAddLog(f"  Read from handle {handle_stim_curr[i]} (I_stim, {bool(sim.samples(handle_stim_curr[i]))})")
		else: # get total membrane current
			data_tot_curr, _ = sim.samples(handle_tot_curr[i])[loc]
			data_curr.append(np.negative(data_tot_curr[:, 1]))
			writeAddLog(f"  Read from handle {handle_tot_curr[i]} (I_tot, {bool(sim.samples(handle_tot_curr[i]))})")

		# get synaptic data
		if sample_syn >= 0: # synapse handles exist only if this condition is true
			if len(sim.samples(handle_Ca_specific[i-dimin])) > 0: 
				data, _ = sim.samples(handle_Ca_specific[i-dimin])[loc]
				data_Ca.append(data[:, 1])
			else:
				data_Ca.append(np.zeros(len(times)))
				writeLog(f"  No data for handle_Ca_specific[{i-dimin}] = {handle_Ca_specific[i-dimin]}")

			if len(sim.samples(handle_h_specific[i-dimin])) > 0: 
				data, _ = sim.samples(handle_h_specific[i-dimin])[loc]
				data_h.append(data[:, 1])
			else:
				data_h.append(np.zeros(len(times)))
				writeLog(f"  No data for handle_h_specific[{i-dimin}] = {handle_h_specific[i-dimin]}")

			if len(sim.samples(handle_z_specific[i-dimin])) > 0: 
				data, _ = sim.samples(handle_z_specific[i-dimin])[loc]
				data_z.append(data[:, 1])
			else:
				data_z.append(np.zeros(len(times)))
				writeLog(f"  No data for handle_z_specific[{i-dimin}] = {handle_z_specific[i-dimin]}")
			
			if len(sim.samples(handle_sps_specific[i-dimin])) > 0: 		
				data, _ = sim.samples(handle_sps_specific[i-dimin])[loc]
				data_sps.append(data[:, 1])
			else:
				data_sps.append(np.zeros(len(times)))
				writeLog(f"  No data for handle_sps_specific[{i-dimin}] = {handle_sps_specific[i-dimin]}")

			if len(sim.samples(handle_p_specific[i-dimin])) > 0: 		
				data, _ = sim.samples(handle_p_specific[i-dimin])[loc]
				data_p.append(data[:, 1])
			else:
				data_p.append(np.zeros(len(times)))
				writeLog(f"  No data for handle_p_specific[{i-dimin}] = {handle_p_specific[i-dimin]}")

		else: # no synaptic handles
			data_Ca.append(np.zeros(len(times)))
			data_h.append(np.zeros(len(times)))
			data_z.append(np.zeros(len(times)))
			data_sps.append(np.zeros(len(times)))
			data_p.append(np.zeros(len(times)))
			dimin += 1 # diminish the following indices by one because of missing synaptic handles
	###########################################################
	# finalize

	# get spikes through 'sim.spikes()' which returns 'structured' NumPy array
	spikes = np.column_stack((sim.spikes()['time'], sim.spikes()['source']['gid']))

	# free memory allocated for the simulation, collect garbage	
	del sim
	gc.collect()

	#breakpoint()
	return spikes, (times, data_mem, data_curr, data_Ca, data_h, data_z, data_sps, data_p)

#####################################
# arborNetworkConsolidation
# Runs simulation of a recurrent neural network with consolidation dynamics
# - config: configuration of model and simulation parameters (as a dictionary from JSON format)
# - add_info [optional]: additional information to be logged
# - platform [optional]: the hardware platform to be used (options: "CPU", "GPU")
# - plotting [optional]: specifies whether plots should be created or not
# - return: the recipe that has been used for the first phase (needed for testing)
def arborNetworkConsolidation(config, add_info = None, platform = "", plotting = False):

	#####################################
	# get parameters that are necessary here
	s_desc = config['simulation']['short_description'] # short description of the simulation
	runtime = config['simulation']['runtime']*U.s # total biological runtime of the simulation (all parts)
	std_dt = config['simulation']['dt']*U.ms # duration of one timestep for rich computation in ms
	ff_dt = config['simulation']['dt_ff']*U.ms # duration of one timestep for fast-forward computation in ms
	sample_gid_list = config['simulation']['sample_gid_list'] # list of the neurons that are to be probed (given by number/gid)
	sample_syn_list = config['simulation']['sample_syn_list'] # list of the synapses that are to be probed (one synapse per neuron only, given by its internal number respective to a neuron in sample_gid); -1: none
	learn_prot = completeProt(config['simulation']['learn_protocol']) # protocol for learning stimulation as a dictionary with the keys "time" (starting time), "scheme" (scheme of pulses), and "freq" (stimulation frequency)
	recall_prot = completeProt(config['simulation']['recall_protocol']) # protocol for recall stimulation as a dictionary with the keys "time" (starting time), "scheme" (scheme of pulses), and "freq" (stimulation frequency)
	bg_prot = completeProt(config['simulation']['bg_protocol']) # protocol for background input as a dictionary with the keys "time" (starting time), "scheme" (scheme of pulses), "I_0" (mean), and "sigma_WN" (standard deviation)
	rich_comp_window = 5000*U.ms # time window of rich computation to consider before and after learning and recall stimulation

	if len(sample_gid_list) != len(sample_syn_list) and len(sample_syn_list) > 1:
		raise ValueError("List of sample synapses has to either have the same length as the list of sample neurons, or to contain maximally one element.")
	elif len(sample_syn_list) == 0:
		sample_syn_list.append(-1)

	#####################################
	# initialize structures to store the data
	dummy_list = len(sample_gid_list) * [np.array([])] #np.ndarray((len(sample_gid_list),))	
	spike_times1, (times1, data_mem1, data_curr1, data_Ca1, data_h1, data_z1, data_sps1, data_p1) = np.ndarray((0,2)), ([], dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list)
	spike_times2, (times2, data_mem2, data_curr2, data_Ca2, data_h2, data_z2, data_sps2, data_p2) = np.ndarray((0,2)), ([], dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list)
	spike_times3, (times3, data_mem3, data_curr3, data_Ca3, data_h3, data_z3, data_sps3, data_p3) = np.ndarray((0,2)), ([], dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list, dummy_list)

	#####################################
	# create output directory, save code and config, and open log file
	out_path = getDataPath(s_desc, refresh=True)
	if not os.path.isdir(out_path): # if the directory does not exist yet
		os.mkdir(out_path)
	os.system("cp -r *.py  \"" + out_path + "\"") # archive the Python code
	os.system("cp -r mechanisms/ \"" + out_path + "\"") # archive the mechanism code
	config['arbor'] = arbor.config() # store the current Arbor core configuation
	json.dump(config, open(getDataPath(s_desc, "config.json"), "w"), indent="\t")
	runf = open(os.path.join(getDataPath(s_desc), "run"), "w") # write a run script
	runf.write(f"#!/bin/sh\n\npython3 ./arborNetworkConsolidation.py -platform=\"{platform}\" -config_file=\"{getTimestamp() + '_config.json'}\"")
	runf.close()
	#global logf # global handle to the log file (to need less code for output commands)
	#logf = open(getDataPath(s_desc, "log.txt"), "w")
	openLog(s_desc)
	verf = open(getDataPath(s_desc, "script_version.txt"), "w")
	verf.write(subprocess.check_output("git show --name-status", shell=True).decode())
	verf.close()
	
	#####################################
	# output of key parameters
	writeLog(f"\x1b[31mArbor network simulation {getTimestamp()} (Arbor version: {arbor.__version__})\n" +
	          "|\n"
	         f"\x1b[35mSimulated timespan:\x1b[37m {runtime} s\n" +
	         f"\x1b[35mPopulation parameters:\x1b[37m N_exc = {config['populations']['N_exc']}, N_inh = {config['populations']['N_inh']}\n" +
	         f"\x1b[35mLearning protocol:\x1b[37m {protFormatted(learn_prot)}\n" +
	         f"\x1b[35mRecall protocol:\x1b[37m {protFormatted(recall_prot)}\n" +
	         f"\x1b[35mBackground input:\x1b[37m {protFormatted(bg_prot)}\x1b[0m\n\x1b[35m" +
	          "|\x1b[0m")
	         
	#####################################
	# start taking the total time and compute random seed
	t_0, _ = getTimeDiff()
	random_seed = int(t_0*10000) # get random seed from system clock
	writeLog(f"Random seed for mechanisms: {random_seed}")
	writeLog(f"Add. info.: {add_info}")
	flushLog()

	###############################################
	# determine beginning and end of stimulation and sampling phases

	# begin and end of stimulation in learning phase
	_, learn_prot_start, learn_prot_end = getInputProtocol(learn_prot, runtime, std_dt,
	                                                       np.array(learn_prot["explicit_input"]["stim_times"]))

	# begin and end of stimulation in recall phase
	_, recall_prot_start, recall_prot_end = getInputProtocol(recall_prot, runtime, std_dt,
	                                                         np.array(recall_prot["explicit_input"]["stim_times"]))

	# begin of sampling for learning phase
	rich_comp_start_earliest = learn_prot_start - rich_comp_window if not np.isnan(learn_prot_start.value_as(U.ms)) else 0*U.ms
	if rich_comp_start_earliest.value_as(U.ms) <= 0:
		learn_sampling_start = 0*U.ms
	else:
		learn_sampling_start = rich_comp_start_earliest

	# end of sampling for learning phase
	rich_comp_end_latest = learn_prot_end + rich_comp_window if not np.isnan(learn_prot_end.value_as(U.ms)) else 0*U.ms
	if np.isnan(learn_prot_end.value_as(U.ms)):
		learn_sampling_end = np.nan*U.ms
	elif rich_comp_end_latest.value_as(U.ms) > runtime.value_as(U.ms):
		learn_sampling_end = runtime
	else:
		learn_sampling_end = rich_comp_end_latest

	# begin of sampling for recall phase
	rich_comp_start_earliest = recall_prot_start - rich_comp_window if not np.isnan(recall_prot_start.value_as(U.ms)) else 0*U.ms
	if rich_comp_start_earliest.value_as(U.ms) <= 0:
		recall_sampling_start = 0*U.ms
	else:
		recall_sampling_start = rich_comp_start_earliest
	
	# end of sampling for recall phase
	rich_comp_end_latest = recall_prot_end + rich_comp_window if not np.isnan(recall_prot_end.value_as(U.ms)) else 0*U.ms
	if np.isnan(recall_prot_end.value_as(U.ms)):
		recall_sampling_end = np.nan*U.ms
	elif rich_comp_end_latest.value_as(U.ms) > runtime.value_as(U.ms):
		recall_sampling_end = runtime
	else:
		recall_sampling_end = rich_comp_end_latest

	# begin of sampling for consolidation phase
	cons_sampling_start = learn_sampling_end if not np.isnan(learn_prot_start.value_as(U.ms)) else 0*U.ms

	# end of sampling for consolidation phase
	cons_sampling_end = recall_sampling_start if not np.isnan(recall_prot_start.value_as(U.ms)) else runtime	

	###############################################
	# initializer functions for simulation phases

	# learning
	def initLearnPhase(phase, store_latest_state):
		adap_dt = std_dt # set the adapted timestep
		writeLog("|")
		writeLog(f"Learning phase (phase {phase})")

		# disable recall protocol
		config_new = copy.deepcopy(config)
		config_new["simulation"]["recall_protocol"] = completeProt({ })

		# use the same connectivity matrix as in previous phases
		if phase > 1:
			config_new['populations']['conn_file'] = getDataPath(s_desc, "connections.txt")

		# set up recipe with according timestep and plasticity mechanism
		recipe = NetworkRecipe(config_new, adap_dt, "expsyn_curr_early_late_plasticity", store_latest_state)

		# set runtime
		t_final = learn_sampling_end
		t_final_red = learn_sampling_end	

		# output of general information
		writeLog("t_final =", t_final.value_as(U.ms), "ms") # final simulated time in ms (effective)
		writeLog("t_final_red =", t_final_red.value_as(U.ms), "ms") # final simulated time in ms (reduced for the current phase)
		writeLog("dt =", adap_dt.value_as(U.ms), "ms") # adapted timestep in ms
		writeLog(f"Learning stim. [begin, end]: [{learn_prot_start.value_as(U.ms)}, {learn_prot_end.value_as(U.ms)}] ms")
		writeLog(f"Learning sampling [begin, end]: [{learn_sampling_start.value_as(U.ms)}, {learn_sampling_end.value_as(U.ms)}] ms")

		return recipe, config_new, t_final, t_final_red, adap_dt

	# consolidation
	def initConsolidationPhase(phase, store_latest_state):
		adap_dt = ff_dt # set the adapted timestep
		writeLog("|")
		writeLog(f"Consolidation phase (phase {phase})")
		
		# disable learning protocol, recall protocol, and background noise
		config_new = copy.deepcopy(config)
		config_new["simulation"]["learn_protocol"] = completeProt({ })
		config_new["simulation"]["recall_protocol"] = completeProt({ })
		config_new["simulation"]["bg_protocol"] = completeProt({ })

		# use the same connectivity matrix as in previous phases
		if phase > 1:
			config_new['populations']['conn_file'] = getDataPath(s_desc, "connections.txt")

		# set up recipe with according timestep and plasticity mechanism
		config_new["simulation"]["output_period"] = 1 # sample every timestep (because they are very large, anyway)
		writeLog(f"output_period_updated = {config_new['simulation']['output_period']}")
		recipe = NetworkRecipe(config_new, adap_dt, "expsyn_curr_early_late_plasticity_ff", store_latest_state)

		# set runtime
		t_final = cons_sampling_end
		t_final_red = cons_sampling_end - cons_sampling_start

		# load previous state, if it exists
		if not np.isnan(learn_sampling_end.value_as(U.ms)):
			recipe.loadState(phase-1)

		# output of general information
		writeLog("t_final =", t_final.value_as(U.ms), "ms") # final simulated time in ms (effective)
		writeLog("t_final_red =", t_final_red.value_as(U.ms), "ms") # final simulated time in ms (reduced for the current phase)
		writeLog("dt =", adap_dt.value_as(U.ms), "ms") # adapted timestep in ms
		writeLog(f"Fast-forward sampling [begin, end]: [{cons_sampling_start.value_as(U.ms)}, {cons_sampling_end.value_as(U.ms)}] ms")

		return recipe, config_new, t_final, t_final_red, adap_dt

	# recall
	def initRecallPhase(phase, store_latest_state):
		adap_dt = std_dt # set the adapted timestep
		writeLog("|")
		writeLog(f"Recall phase (phase {phase})")

		# disable learning protocol
		config_new = copy.deepcopy(config)
		config_new["simulation"]["learn_protocol"] = completeProt({ })		

		# use the same connectivity matrix as in previous phases
		if phase > 1:
			config_new['populations']['conn_file'] = getDataPath(s_desc, "connections.txt")

		# set up recipe with according timestep and plasticity mechanism
		recipe = NetworkRecipe(config_new, adap_dt, "expsyn_curr_early_late_plasticity", store_latest_state)

		# set runtime
		t_final = recall_sampling_end
		t_final_red = recall_sampling_end - recall_sampling_start

		# load previous state (if it exists)
		if recall_sampling_start.value_as(U.ms) > 0:
			recipe.loadState(phase-1)

		# adjust recall protocol start by effective final time of previous phase (subtracting one timestep to prevent limit case errors)
		recipe.recall_prot["time_start"] -= (recall_sampling_start.value_as(U.s) - std_dt.value_as(U.s)) 

		# output of general information
		writeLog("t_final =", t_final.value_as(U.ms), "ms") # final simulated time in ms (effective)
		writeLog("t_final_red =", t_final_red.value_as(U.ms), "ms") # final simulated time in ms (reduced for the current phase)
		writeLog("dt =", adap_dt.value_as(U.ms), "ms") # adapted timestep in ms
		writeLog(f"Recall stim. [begin, end]: [{recall_prot_start.value_as(U.ms)}, {recall_prot_end.value_as(U.ms)}] ms")
		writeLog(f"Recall sampling [begin, end]: [{recall_sampling_start.value_as(U.ms)}, {recall_sampling_end.value_as(U.ms)}] ms")

		return recipe, config_new, t_final, t_final_red, adap_dt

	# learning + recall
	def initLearnRecallPhase(phase, store_latest_state):
		adap_dt = std_dt # set the adapted timestep
		writeLog("|")
		writeLog(f"Learning+Recall phase (phase {phase})")

		# create a copy of config
		config_new = copy.deepcopy(config)

		# use the same connectivity matrix as in previous phases
		if phase > 1:
			config_new['populations']['conn_file'] = getDataPath(s_desc, "connections.txt")

		# set up recipe with according timestep and plasticity mechanism
		recipe = NetworkRecipe(config_new, adap_dt, "expsyn_curr_early_late_plasticity", store_latest_state)

		# set runtime
		t_final = recall_sampling_end
		t_final_red = recall_sampling_end

		# output of general information
		writeLog("t_final =", t_final.value_as(U.ms), "ms") # final simulated time in ms (effective)
		writeLog("t_final_red =", t_final_red.value_as(U.ms), "ms") # final simulated time in ms (reduced for the current phase)
		writeLog("dt =", adap_dt.value_as(U.ms), "ms") # adapted timestep in ms
		writeLog(f"Learning stim. [begin, end]: [{learn_prot_start.value_as(U.ms)}, {learn_prot_end.value_as(U.ms)}] ms")
		writeLog(f"Learning sampling [begin, end]: [{learn_sampling_start.value_as(U.ms)}, {learn_sampling_end.value_as(U.ms)}] ms")
		writeLog(f"Recall stim. [begin, end]: [{recall_prot_start.value_as(U.ms)}, {recall_prot_end.value_as(U.ms)}] ms")
		writeLog(f"Recall sampling [begin, end]: [{recall_sampling_start.value_as(U.ms)}, {recall_sampling_end.value_as(U.ms)}] ms")

		return recipe, config_new, t_final, t_final_red, adap_dt

	# background (NOTE: not compatible with other phases!)
	def initBackgroundPhase(phase, store_latest_state):
		adap_dt = std_dt # set the adapted timestep
		writeLog("|")
		writeLog(f"Background phase (phase {phase})")

		# disable learning protocol and recall protocol
		config_new = copy.deepcopy(config)
		config_new["simulation"]["learn_protocol"] = completeProt({ })
		config_new["simulation"]["recall_protocol"] = completeProt({ })

		# set up recipe with according timestep and plasticity mechanism
		recipe = NetworkRecipe(config_new, adap_dt, "expsyn_curr_early_late_plasticity", store_latest_state)

		# set runtime
		t_final = runtime
		t_final_red = runtime

		# output of general information
		writeLog("t_final =", t_final.value_as(U.ms), "ms") # final simulated time in ms (effective)
		writeLog("t_final_red =", t_final_red.value_as(U.ms), "ms") # final simulated time in ms (reduced for the current phase)
		writeLog("dt =", adap_dt.value_as(U.ms), "ms") # adapted timestep in ms

		return recipe, config_new, t_final, t_final_red, adap_dt

	###############################################
	# schedule of simulation phases
	# -> 8 possibilities for a schedule with learning phase, consolidation phase, recall phase
	#    (if none of the phases is given, there may still be background activity)

	# learn - consolidate - recall
	if not np.isnan(learn_prot_start.value_as(U.ms)) and \
	   not np.isnan(recall_prot_start.value_as(U.ms)) and \
	   learn_sampling_end.value_as(U.ms) < recall_sampling_start.value_as(U.ms):

		writeLog("Schedule: learn - consolidate - recall")
		
		# set up and run simulation phase 1: learning
		recipe_1, config_new, t_final_1, t_final_red, adap_dt = initLearnPhase(1, True)
		spike_times1, (times1, data_mem1, data_curr1, data_Ca1, data_h1, data_z1, data_sps1, data_p1) \
		  = runSimPhase(1, recipe_1, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_1, t_diff = getTimeDiff(t_0)
		writeLog("Phase 1 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		# set up and run simulation phase 2: consolidation
		recipe_2, config_new, t_final_2, t_final_red, adap_dt = initConsolidationPhase(2, True)
		spike_times2, (times2, data_mem2, data_curr2, data_Ca2, data_h2, data_z2, data_sps2, data_p2) \
		  = runSimPhase(2, recipe_2, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_2, t_diff = getTimeDiff(t_1)
		writeLog("Phase 2 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		# set up and run simulation phase 3: recall
		recipe_3, config_new, t_final_3, t_final_red, adap_dt = initRecallPhase(3, False)
		spike_times3, (times3, data_mem3, data_curr3, data_Ca3, data_h3, data_z3, data_sps3, data_p3) \
		 = runSimPhase(3, recipe_3, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_3, t_diff = getTimeDiff(t_2)
		writeLog("Phase 3 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		recipe_to_return = recipe_3

	# learn - recall
	# (sampling phases of learning and recall overlap -> no consolidation)
	elif not np.isnan(learn_prot_start.value_as(U.ms)) and \
	     not np.isnan(recall_prot_start.value_as(U.ms)) and \
	     learn_sampling_end.value_as(U.ms) >= recall_sampling_start.value_as(U.ms):

		writeLog("Schedule: learn+recall")

		# trim sampling periods
		learn_sampling_end = learn_prot_end
		recall_sampling_start = learn_prot_end

		# set up and run simulation phase 1: learning
		recipe_1, config_new, t_final_1, t_final_red, adap_dt = initLearnRecallPhase(1, False)
		spike_times1, (times1, data_mem1, data_curr1, data_Ca1, data_h1, data_z1, data_sps1, data_p1) \
		  = runSimPhase(1, recipe_1, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_1, t_diff = getTimeDiff(t_0)
		writeLog("Phase 1 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		t_final_3 = t_final_2 = t_final_1
		t_3 = t_2 = t_1

		recipe_to_return = recipe_1

	# learn - consolidate
	# (sampling phase of learning is shorter than total runtime; no recall protocol)
	elif not np.isnan(learn_prot_start.value_as(U.ms)) and \
	     np.isnan(recall_prot_start.value_as(U.ms)) and \
	     learn_sampling_end.value_as(U.ms) < runtime.value_as(U.ms):	

		writeLog("Schedule: learn - consolidate")

		# set up and run simulation phase 1: learning
		recipe_1, config_new, t_final_1, t_final_red, adap_dt = initLearnPhase(1, True)
		spike_times1, (times1, data_mem1, data_curr1, data_Ca1, data_h1, data_z1, data_sps1, data_p1) \
		  = runSimPhase(1, recipe_1, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_1, t_diff = getTimeDiff(t_0)
		writeLog("Phase 1 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		# set up and run simulation phase 2: consolidation
		recipe_2, config_new, t_final_2, t_final_red, adap_dt = initConsolidationPhase(2, False)
		spike_times2, (times2, data_mem2, data_curr2, data_Ca2, data_h2, data_z2, data_sps2, data_p2) \
		  = runSimPhase(2, recipe_2, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_2, t_diff = getTimeDiff(t_1)
		writeLog("Phase 2 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		t_final_3 = t_final_2
		t_3 = t_2

		recipe_to_return = recipe_2

	# learn
	# (sampling phase of learning reaches total runtime; no recall protocol)
	elif not np.isnan(learn_prot_start.value_as(U.ms)) and \
	     np.isnan(recall_prot_start.value_as(U.ms)) and \
	     learn_sampling_end.value_as(U.ms) == runtime.value_as(U.ms):	

		writeLog("Schedule: learn")

		# set up and run simulation phase 1: learning
		recipe_1, config_new, t_final_1, t_final_red, adap_dt = initLearnPhase(1, False)
		spike_times1, (times1, data_mem1, data_curr1, data_Ca1, data_h1, data_z1, data_sps1, data_p1) \
		  = runSimPhase(1, recipe_1, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_1, t_diff = getTimeDiff(t_0)
		writeLog("Phase 1 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		t_final_3 = t_final_2 = t_final_1
		t_3 = t_2 = t_1

		recipe_to_return = recipe_1

	# consolidate - recall
	# (no learning protocol; sampling phase of recall starts after the simulation has started)
	elif np.isnan(learn_prot_start.value_as(U.ms)) and \
	     not np.isnan(recall_prot_start.value_as(U.ms)) and \
	     recall_sampling_start.value_as(U.ms) > 0:	

		writeLog("Schedule: consolidate - recall")

		# set up and run simulation phase 1: consolidation
		recipe_1, config_new, t_final_1, t_final_red, adap_dt = initConsolidationPhase(1, True)
		spike_times1, (times1, data_mem1, data_curr1, data_Ca1, data_h1, data_z1, data_sps1, data_p1) \
		  = runSimPhase(1, recipe_1, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_1, t_diff = getTimeDiff(t_0)
		writeLog("Phase 1 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		# set up and run simulation phase 2: recall
		recipe_2, config_new, t_final_2, t_final_red, adap_dt = initRecallPhase(2, False)
		spike_times2, (times2, data_mem2, data_curr2, data_Ca2, data_h2, data_z2, data_sps2, data_p2) \
		  = runSimPhase(2, recipe_2, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_2, t_diff = getTimeDiff(t_1)
		writeLog("Phase 2 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		t_final_3 = t_final_2
		t_3 = t_2

		recipe_to_return = recipe_2

	# recall
	# (no learning protocol; sampling phase of recall starts when the simulation starts)
	elif np.isnan(learn_prot_start.value_as(U.ms)) and \
	     not np.isnan(recall_prot_start.value_as(U.ms)) and \
	     recall_sampling_start.value_as(U.ms) == 0:

		writeLog("Schedule: recall")

		# set up and run simulation phase 1: recall
		recipe_1, config_new, t_final_1, t_final_red, adap_dt = initRecallPhase(1, False)
		spike_times1, (times1, data_mem1, data_curr1, data_Ca1, data_h1, data_z1, data_sps1, data_p1) \
		  = runSimPhase(1, recipe_1, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_1, t_diff = getTimeDiff(t_0)
		writeLog("Phase 1 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		t_final_3 = t_final_2 = t_final_1
		t_3 = t_2 = t_1

		recipe_to_return = recipe_1

	# consolidate (only fast-forward computation)
	# (no learning protocol; no recall protocol; simulation longer than '5*rich_comp_window')
	elif np.isnan(learn_prot_start.value_as(U.ms)) and \
	     np.isnan(recall_prot_start.value_as(U.ms)) and \
	     runtime.value_as(U.ms) >= 5*rich_comp_window.value_as(U.ms):

		writeLog("Schedule: consolidate")

		# set up and run simulation phase 1: consolidation
		recipe_1, config_new, t_final_1, t_final_red, adap_dt = initConsolidationPhase(1, False)
		spike_times1, (times1, data_mem1, data_curr1, data_Ca1, data_h1, data_z1, data_sps1, data_p1) \
		  = runSimPhase(1, recipe_1, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_1, t_diff = getTimeDiff(t_0)
		writeLog("Phase 1 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		t_final_3 = t_final_2 = t_final_1
		t_3 = t_2 = t_1

		recipe_to_return = recipe_1

	# background (only rich computation)
	# (no learning protocol; no recall protocol; simulation shorter than '5*rich_comp_window')
	elif np.isnan(learn_prot_start.value_as(U.ms)) and \
	     np.isnan(recall_prot_start.value_as(U.ms)) and \
	     runtime.value_as(U.ms) < 5*rich_comp_window.value_as(U.ms):

		writeLog("Schedule: background")

		# set up and run simulation phase 1: background
		recipe_1, config_new, t_final_1, t_final_red, adap_dt = initBackgroundPhase(1, False)
		spike_times1, (times1, data_mem1, data_curr1, data_Ca1, data_h1, data_z1, data_sps1, data_p1) \
		  = runSimPhase(1, recipe_1, config_new, random_seed, t_final_red, adap_dt, platform, plotting)
		t_1, t_diff = getTimeDiff(t_0)
		writeLog("Phase 1 completed (in " + getFormattedTime(round(t_diff)) + ").")
		writeLog("|")

		t_final_3 = t_final_2 = t_final_1
		t_3 = t_2 = t_1

		recipe_to_return = recipe_1

	else:
		raise Exception(f"An error occurred when assessing the simulation schedule" + \
		                f"(learn_prot_start = {learn_prot_start}, " + \
		                f"recall_prot_start = {recall_prot_start}, " + \
		                f"learn_sampling_end = {learn_sampling_end}, " + \
		                f"recall_sampling_start = {recall_sampling_start}).")

	#####################################
	# re-normalize spike times and trace times from phases 2 and 3
	if t_final_2.value_as(U.ms) > t_final_1.value_as(U.ms):
		spike_times2[:,0] += t_final_1.value_as(U.ms)
		if len(sample_gid_list) > 0:
			times2 += t_final_1.value_as(U.ms)

	if t_final_3.value_as(U.ms) > t_final_2.value_as(U.ms):
		spike_times3[:,0] += t_final_2.value_as(U.ms)
		if len(sample_gid_list) > 0:
			times3 += t_final_2.value_as(U.ms)
	
	#####################################
	# information on data obtained from the phases
	writeLog("Spikes in phase #1:", len(spike_times1[:,0]))
	writeLog("Spikes in phase #2:", len(spike_times2[:,0]))
	writeLog("Spikes in phase #3:", len(spike_times3[:,0]))
	writeLog("Datapoints in phase #1:", len(times1))
	writeLog("Datapoints in phase #2:", len(times2))
	writeLog("Datapoints in phase #3:", len(times3))

	#####################################
	# assemble and store data
	spike_times = np.vstack((spike_times1, spike_times2, spike_times3))
	times = [*times1, *times2, *times3]

	data_header = "Time, "
	data_stacked = times # start with times, then add the data columns in the right order	

	# loop over elements in 'sample_gid_list' (and thereby, 'sample_syn_list') to assemble the retrieved data
	for i in range(len(sample_gid_list)):
		# retrieve the current neuron index
		sample_gid = sample_gid_list[i]

		# retrieve the synapse index by the corresponding neuron identifier
		sample_syn = getSynapseId(sample_syn_list, i)

		# retrieve the synapse index by finding out how often the current neuron index has occurred already 
		sample_syn_alt = sample_gid_list[:i].count(sample_gid)

		writeAddLog(f"Assembling data for sample #{i} " + 
		            f"(sample_gid = {sample_gid}, sample_syn = {sample_syn}, sample_syn_alt = {sample_syn_alt})...")

		# prepare strings for header output
		neur = str(sample_gid)
		syn = str(sample_syn)

		datacol_mem = np.hstack((data_mem1[i], data_mem2[i], data_mem3[i]))
		datacol_curr = np.hstack((data_curr1[i], data_curr2[i], data_curr3[i]))
		datacol_h = np.hstack((data_h1[i], data_h2[i], data_h3[i]))
		datacol_z = np.hstack((data_z1[i], data_z2[i], data_z3[i]))*recipe_to_return.h_0.value_as(U.mV)   # normalize by initial weight to get the same
		                                                                                                  # unit as for `h`
		datacol_Ca = np.hstack((data_Ca1[i], data_Ca2[i], data_Ca3[i]))
		datacol_spsV = np.hstack((data_sps1[i], data_sps2[i], data_sps3[i]))  # no need to multiply by V because the values 
		                                                                      # from the mechanisms are already normalized
		datacol_p = np.hstack((data_p1[i], data_p2[i], data_p3[i]))

		# stack the data
		data_header += f"V({neur}), I({neur}), "
		data_stacked = np.column_stack([data_stacked,
		                                datacol_mem, datacol_curr])

		data_header += f"h({neur},,{syn}), z({neur},,{syn}), Ca({neur},,{syn}), " + \
				       f"spsV({neur},,{syn}), p^C({neur},,{syn}), "
		data_stacked = np.column_stack([data_stacked,
				                        datacol_h, datacol_z, datacol_Ca, datacol_spsV, datacol_p])

	np.savetxt(getDataPath(s_desc, "traces.txt"), data_stacked, fmt="%.4f", header=data_header)
	np.savetxt(getDataPath(s_desc, "spikes.txt"), spike_times, fmt="%.4f %.0f") # integer formatting for neuron number
	t_4, _ = getTimeDiff()
	writeAddLog("Data stored (in " + getFormattedTime(round(t_4 - t_3)) + ").")
	
	#####################################
	# do the plotting
	if plotting:
		from plotResults import plotResults, plotRaster

		for i in range(len(sample_gid_list)):
			# retrieve the current neuron index
			sample_gid = sample_gid_list[i]

			# retrieve the synapse index by the corresponding neuron identifier
			sample_syn = getSynapseId(sample_syn_list, i)

			# retrieve the synapse index by finding out how often the current neuron index has occurred already 
			#sample_syn_alt = sample_gid_list[:i].count(sample_gid)

			# call the plot function for neuron and synapse traces, depending on whether reduced or all data shall be plotted
			plotResults(config, data_stacked, getTimestamp(), i, mem_dyn_data = True, neuron=sample_gid, synapse=sample_syn, store_path=getDataPath(s_desc), figure_fmt = 'svg')
		
		# call the plot function for spike raster	
		plotRaster(config, spike_times.T, getTimestamp(), store_path=getDataPath(s_desc), figure_fmt = 'png')
		t_5, _ = getTimeDiff()
		writeAddLog("Plotting completed (in " + getFormattedTime(round(t_5 - t_4)) + ").")
	else:
		t_5 = t_4
	#####################################
    # close the log file
	writeLog("Total elapsed time: " + getFormattedTime(round(t_5 - t_0)) + ".")
	closeLog()

	return recipe_to_return

#####################################
if __name__ == '__main__':
	
	# parse the commandline parameter 'config_file'
	parser = argparse.ArgumentParser()
	parser.add_argument('-config_file', required=True, help="configuration of the simulation parameters (JSON file)")
	(args, unknown) = parser.parse_known_args()
	
	# load JSON object containing the parameter configuration as dictionary
	config = json.load(open(args.config_file, "r"))
	
	# parse the remaining commandline parameters
	parser.add_argument('-s_desc', type=str, help="short description")
	parser.add_argument('-add_info', type=str, help="additional information to be logged")
	parser.add_argument('-platform', type=str, help="the hardware platform to be used")
	parser.add_argument('-runtime', type=float, help="runtime of the simulation in s")
	parser.add_argument('-dt', type=float, help="duration of one timestep in ms")
	parser.add_argument('-t_ref', type=float, help="refractory period in ms")
	parser.add_argument('-output_period', type=int, help="sampling size in timesteps")
	parser.add_argument('-conn', help="file containing a connectivity matrix")
	parser.add_argument('-N_CA', type=int, help="number of neurons in the cell assembly")
	parser.add_argument('-learn', type=str, help="protocol for learning stimulation")
	parser.add_argument('-recall', type=str, help="protocol for recall stimulation")
	parser.add_argument('-sample_gid', type=int, nargs='+', help="numbers of the neurons that shall be probed")
	parser.add_argument('-sample_syn', type=int, nargs='+', help="relative numbers of the synapses that shall be probed")
	parser.add_argument('-sample_curr', type=int, help="current that shall be probed")
	parser.add_argument('-w_ei', type=float, help="excitatory to inhibitory coupling strength in units of h_0")
	parser.add_argument('-w_ie', type=float, help="inhibitory to excitatory coupling strength in units of h_0")
	parser.add_argument('-w_ii', type=float, help="inhibitory to excitatory coupling strength in units of h_0")
	parser.add_argument('-plot', type=int, help="switch plotting on/off")
	args = parser.parse_args()

	# in the dictionary containing the parameter configuration, modify the values provided by commandline arguments
	if (args.s_desc is not None): config['simulation']['short_description'] = args.s_desc
	if (args.runtime is not None): config['simulation']['runtime'] = args.runtime
	if (args.dt is not None): config['simulation']['dt'] = args.dt
	if (args.t_ref is not None): config['neuron']['t_ref'] = args.t_ref
	if (args.output_period is not None): config['simulation']['output_period'] = args.output_period
	if (args.conn is not None): config['populations']['conn_file'] = args.conn
	if (args.N_CA is not None): config['populations']['N_CA'] = args.N_CA
	if (args.learn is not None): config['simulation']['learn_protocol'] = json.loads(args.learn) # convert "learn" argument string into dictionary
	if (args.recall is not None): config['simulation']['recall_protocol'] = json.loads(args.recall) # convert "recall" argument string into dictionary
	if (args.sample_gid is not None): config['simulation']['sample_gid_list'] = args.sample_gid
	if (args.sample_syn is not None): config['simulation']['sample_syn_list'] = args.sample_syn
	if (args.sample_curr is not None): config['simulation']['sample_curr'] = args.sample_curr
	if (args.w_ei is not None): config['populations']['w_ei'] = args.w_ei
	if (args.w_ie is not None): config['populations']['w_ie'] = args.w_ie
	if (args.w_ii is not None): config['populations']['w_ii'] = args.w_ii

	# assign the other arguments
	add_info = args.add_info if (args.add_info is not None) else ""
	platform = args.platform if (args.platform is not None) else "CPU" # choose "CPU" if no platform is specified
	plotting = bool(args.plot) if (args.plot is not None) else True # do plotting if nothing is specified

	# run the simulation
	try:
		arborNetworkConsolidation(config, add_info, platform, plotting)
	except:
		errf = open(getDataPath(config['simulation']['short_description'], "errorlog.txt"), "w")
		traceback.print_exc(file=errf)
		errf.close()
		traceback.print_exc()
	
