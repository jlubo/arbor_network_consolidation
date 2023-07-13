: Delta synapse without plasticity

NEURON {
	POINT_PROCESS deltasyn
	RANGE g_spike
	NONSPECIFIC_CURRENT I
}

UNITS {
	(ms) = (milliseconds)
	(mV) = (millivolt)
}

PARAMETER {
	g_spike = 1 (mV) : increase in g through one spike
}

ASSIGNED {
}

STATE {
	g (mV) : instantaneous synaptic conductance (non-zero if there a spike in this timestep)
}

INITIAL {
	g = 0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	
	if (g > 0) { : if there is a spike
		I = -g
		g = 0
	}
	else {
		I = 0
	}
}

DERIVATIVE state {
}

NET_RECEIVE(weight) {
	g = g_spike
}

