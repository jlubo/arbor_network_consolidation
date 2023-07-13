: Current-based synapse with exponential postsynatic potential; featuring
: early-phase plasticity (transient) based on the postsynaptic calcium amount, and
: late-phase plasticity (permanent/line attractor) with synaptic tagging and capture

NEURON {
	POINT_PROCESS expsyn_curr_early_late_plasticity
	RANGE h_0, tau_syn, Ca_pre, Ca_post, theta_p, theta_d, theta_tag, R_mem, sigma_pl, h_init, z_init, area
	NONSPECIFIC_CURRENT I
	USEION sps_ WRITE sps_d : signal to trigger protein synthesis
	USEION pc_ READ pc_d    : common pool of plasticity-related proteins
}

UNITS {
	(ms)    = (milliseconds)
	(mV)    = (millivolt)
	(MOhm)  = (megaohm)
}

PARAMETER {
	: RANGE parameters
	h_0         =  4.20075    (mV)   : initial weight
	tau_syn     =  5.0        (ms)   : synaptic time constant
	Ca_pre      =  1.0               : pre-synaptic calcium contribution
	Ca_post     =  0.2758            : post-synaptic calcium contribution
	theta_p     =  3.0               : calcium threshold for potentiation
	theta_d     =  1.2               : calcium threshold for depression
	theta_tag   =  0.840149   (mV)   : tagging threshold
	R_mem       =  10.0       (MOhm) : membrane resistance
	sigma_pl    =  2.90436    (mV)   : standard deviation for plasticity fluctuations

	: GLOBAL parameters
	tau_h       =  688400     (ms)    : early-phase time constant
	tau_z       =  3600000    (ms)    : late-phase time constant
	tau_Ca      =  48.8       (ms)    : calcium time constant
	gamma_p     =  1645.6             : potentiation rate
	gamma_d     =  313.1              : depression rate
	t_Ca_delay  =  18.8       (ms)    : delay of postsynaptic calcium increase after presynaptic spike

	h_init      =  4.20075    (mV)    : parameter to set state variable h
	z_init      =  0                  : parameter to set state variable z

	diam                              : CV diameter in µm (internal variable)
	area                              : CV area of effect in µm^2 (internal variable in newer Arbor versions)
}

STATE {
	psp    (mV)  : instant value of postsynaptic potential
	h      (mV)  : early-phase weight
	z            : late-phase weight
	Ca           : calcium concentration
}

ASSIGNED {
	w (mV)          : total synaptic weight
	h_diff_abs_prev : relative early-phase weight in the previous timestep (absolute value)
	pc              : common pool of plasticity-related proteinss (mmol/l)
}

INITIAL {
	psp  = 0
	h = h_init
	z = z_init
	Ca = 0

	w = h_init + z_init*h_0
	h_diff_abs_prev = 0
	pc = pc_d
}

BREAKPOINT {
	SOLVE state METHOD stochastic
	LOCAL h_diff_abs

	: Set postsynaptic current
	w = h + z*h_0
	I = -psp / R_mem

	: Update the signal triggering protein synthesis
	h_diff_abs = fabs(h - h_0) : current relative early-phase weight (absolute value)
	sps_d = sps_d + (h_diff_abs - h_diff_abs_prev) * area / 1000 : note: not divided by volume, so that it represents the amount
	h_diff_abs_prev = h_diff_abs

	: Retrieve the protein concentration
	pc = pc_d
}

WHITE_NOISE {
	zeta
}

DERIVATIVE state {
	LOCAL xi, ltp_switch, ltd_switch, h_diff
	
	: Exponential decay of postsynaptic potential
	psp' = - psp / tau_syn

	: Relative early-phase weight and other helper variables
    h_diff = h - h_0
	ltp_switch = step_right(Ca - theta_p)
	ltd_switch = step_right(Ca - theta_d)
	xi = sqrt(tau_h * (ltp_switch + ltd_switch)) * sigma_pl * zeta
	
	: Early-phase dynamics
	h' = (- 0.1 * h_diff + gamma_p * (10 - h) * ltp_switch - gamma_d * h * ltd_switch + xi) / tau_h
	
	: Late-phase dynamics
	z' = pc * ((1 - z) * step_right(h_diff - theta_tag) - (z + 0.5) * step_right(- h_diff - theta_tag)) / tau_z
	
	: Exponential decay of calcium concentration
	Ca' = - Ca / tau_Ca
}

NET_RECEIVE(weight) {
	if (weight >= 0) {
		: Start of postsynaptic potential
		psp = psp + w
	}
	else {
		: Increase of calcium amount by presynaptic spike
		Ca = Ca + Ca_pre
	}
}

POST_EVENT(time) {
	: Increase of calcium amount by postsynaptic spike
	Ca = Ca + Ca_post
}

