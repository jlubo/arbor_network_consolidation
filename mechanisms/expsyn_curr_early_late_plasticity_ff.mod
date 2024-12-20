: Like 'expsyn_curr_early_late_plasticity', but for fast-forward computation with late-phase dynamics only

NEURON {
	POINT_PROCESS expsyn_curr_early_late_plasticity_ff
	RANGE h_0, theta_tag, h_init, z_init, area
	NONSPECIFIC_CURRENT I
	USEION pc_ READ pc_d : common pool of plasticity-related proteins
	USEION sps_ WRITE sps_d : signal triggering protein synthesis
}

UNITS {
	(ms)    = (milliseconds)
	(mV)    = (millivolt)
	(MOhm)  = (megaohm)
}

PARAMETER {
	: RANGE parameters
	h_0         =  4.20075    (mV)   : initial weight
	theta_tag   =  0.840149   (mV)   : tagging threshold
	R_mem       =  10.0       (MOhm) : membrane resistance

	: GLOBAL parameters
	tau_h       =  688400     (ms)    : early-phase time constant
	tau_z       =  3600000    (ms)    : late-phase time constant
	f_int       =  0.1                : late-phase integration factor (in l/µmol)

	h_init      =  4.20075    (mV)    : parameter to set state variable h
	z_init      =  0                  : parameter to set state variable z

	diam                              : CV diameter (internal variable, in µm)
	area                              : CV area of effect (internal variable in newer Arbor versions,  in µm^2)
}

STATE {
	h      (mV)     : early-phase weight
	z               : late-phase weight
	Ca              : calcium concentration
	pc              : common pool of plasticity-related proteins (in µmol/l or mmol/l or ...)
}

ASSIGNED {
	w (mV)          : total synaptic weight
	h_diff_abs_prev : relative early-phase weight in the previous timestep (absolute value)
}

INITIAL {
	h = h_init
	z = z_init
	Ca = 0

	h_diff_abs_prev = 0
	pc = pc_d
}

BREAKPOINT {
	:SOLVE state METHOD cnexp : solver not compatible with late-phase equation
	SOLVE state METHOD sparse
	LOCAL h_diff_abs

	I = 0

	: Update the signal triggering protein synthesis
	h_diff_abs = fabs(h - h_0) : current relative early-phase weight (absolute value)
	sps_d = sps_d + (h_diff_abs - h_diff_abs_prev) * area / 1000 : note: not divided by volume, so that it represents the amount
	h_diff_abs_prev = h_diff_abs

	: Retrieve the protein concentration
	pc = pc_d
}

DERIVATIVE state {
	LOCAL h_diff

	: Relative early-phase weight
    h_diff = h - h_0

	: Early-phase decay
	h' = (- 0.1 * h_diff) / tau_h
	
	: Late-phase dynamics
	z' = pc * f_int * ((1 - z) * step_right(h_diff - theta_tag) - (z + 0.5) * step_right(- h_diff - theta_tag)) / tau_z
}

NET_RECEIVE(weight) {

}

POST_EVENT(time) {

}
