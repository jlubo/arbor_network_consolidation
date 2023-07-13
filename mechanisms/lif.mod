: Leaky-integrate and fire (LIF) neuron with protein pool
: based on the LIF mechanism written by Sebastian Schmitt (https://github.com/tetzlab/FIPPA/blob/main/STDP/mechanisms/lif.mod)

NEURON {
	SUFFIX lif
	RANGE R_reset, R_leak, V_reset, V_rev, V_th, I_0, pc_init, i_factor, t_ref, theta_pro
	NONSPECIFIC_CURRENT i
	USEION sps_ READ sps_d : signal to trigger protein synthesis
	USEION pc_ WRITE pc_d : common pool of plasticity-related proteins
}

UNITS {
	(ms)      =  (milliseconds)
	(mV)      =  (millivolt)
	(MOhm)    =  (megaohm)
}

PARAMETER {
	R_leak    =   10       (MOhm) : membrane resistance during leakage (standard case)
	R_reset   =   1e-10    (MOhm) : membrane resistance during refractory period (very small to yield fast reset)
	I_0       =   0               : constant input current (in nA)
	i_factor  =   1               : current to current density conversion factor (nA to mA/cm^2; for point neurons)

	V_reset   =  -70      (mV)   : reset potential
	V_rev     =  -65      (mV)   : reversal potential
	V_th      =  -55      (mV)   : threshold potential to be crossed for spiking
	t_ref     =   2       (ms)   : refractory period

	theta_pro =   2.10037 (mV)   : protein synthesis threshold
	tau_p     =   3600000 (ms)   : protein time constant
	alpha     =   1              : protein synthesis rate (mmol/l)
	
	pc_init   =   0              : parameter to set state variable pc

	:diam                         : CV diameter in µm (internal variable)
	:area                         : CV area of effect in µm^2 (internal variable in newer Arbor versions)
}

:ASSIGNED {
:	volume : volume of the CV (conversion factor between concentration and particle amount, in µm^3)
:}

STATE {
	refractory_counter
	spsV : signal to trigger protein synthesis (particle amount; 1e-18 mol)
	pc : common pool of plasticity-related proteins (mmol/l)
}

INITIAL {
	:volume = area*diam/4 : = area*r/2 = 2*pi*r*h*r/2 = pi*r^2*h
	refractory_counter = t_ref + 1 : start not refractory
	v = V_rev : set the initial membrane potential
	spsV = sps_d
	pc = pc_init
	pc_d = pc_init
}

BREAKPOINT {
	SOLVE state METHOD cnexp

	LOCAL R_mem
	LOCAL E
		
	: threshold crossed -> start refractory counter
	if (v > V_th) {
	   refractory_counter = 0
	}

	: choose between leak and reset potential
	if (refractory_counter <= t_ref) { : in refractory period - strong drive of v towards V_reset
		R_mem = R_reset
		E = V_reset
	} else { : outside refractory period
		R_mem = R_leak
		E = V_rev
	}

	: current density in units of mA/cm^2
	i = (((v - E) / R_mem) - I_0) * i_factor

	: get the signal to trigger protein synthesis (already normalized to represent the amount in the whole cell)
	spsV = sps_d

	: set the new protein concentration (computed from differential equation)
	pc_d = pc
}

DERIVATIVE state {
	if (refractory_counter <= t_ref) {
		refractory_counter' = 1
	}

	pc' = ( -pc + alpha * step_right( spsV - theta_pro ) ) / tau_p
}

