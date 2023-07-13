# Arbor simulation of memory consolidation in recurrent spiking neural networks based on synaptic tagging and capture

This package serves to simulate recurrent spiking neural networks, consisting of leaky integrate-and-fire neurons connected via current-based plastic synapses, with the Arbor simulator library.
The long-term plasticity model that is employed features a calcium-based early phase and a late phase that is based on synaptic tagging-and-capture mechanisms. 
The underlying model has been described in detail in [Luboeinski and Tetzlaff (2021)](https://doi.org/10.1038/s42003-021-01778-y) and has previously been implemented with a [stand-alone simulator](https://github.com/jlubo/memory-consolidation-stc). The code provided here can reproduce the previous results with the Arbor simulator library.
An implementation that employs Arbor to simulate similar networks with morphological neurons will be made available soon.

### Provided code
The main simulation code is found in `arborNetworkConsolidation.py`. The code in `plotResults.py` is used to plot the results and is called automatically after a simulation is finished (but it can also be used on its own). The file `outputUtilities.py` provides additional utility functions needed. The parameter configurations for different types of simulations are provided by means of `*.json` files. The neuron and synapse mechanisms are provided in the [Arbor NMODL format](https://docs.arbor-sim.org/en/v0.8.1/fileformat/nmodl.html) in the files `mechanisms/*.mod`.

To achieve viable runtimes, the program determines a "schedule" for each simulation (e.g., "learn - consolidate"). This causes the simulation to either run in one piece, or to be split up into different phases which are computed using different timesteps (small timesteps for detailed dynamics and substantially longer timesteps for phases of plasticity consolidation). According to this, the plasticity mechanism (`expsyn\_curr\_early\_late\_plasticity` or `expsyn\_curr\_early\_late\_plasticity\_ff`, respectively) is chosen.

Different simulation protocols can easily be run via the following bash script files:
 * `run\_basic\_early` 
   - example of basic early-phase plasticity dynamics in a small network
 * `run\_basic\_late`
   - example of basic late-phase plasticity dynamics in a small network
 * `run\_benchmark\_desktop`
   - pipeline for benchmarks of runtime and memory usage (for the latter, the script `track_allocated_memory` is used); 
   - can be used with different paradigms (`CA200`, `2N1S\_basic\_late`, ...)
 * `run\_defaultnet\_bg\_only\_desktop` 
   - network of 1600 excitatory and 400 inhibitory neurons;
   - background input only
 * `run\_defaultnet\_10s-recall\_desktop`
   - network of 1600 excitatory and 400 inhibitory neurons;
   - learning a memory represented by a cell assembly of a specified number of exc. neurons (_argument $1_);
   - recall of the memory after 10 seconds
 * `run\_defaultnet\_8h-recall\_desktop`
   - network of 1600 excitatory and 400 inhibitory neurons;
   - learning a memory represented by a cell assembly of a specified number of exc. neurons (_argument $1_);
   - consolidation via synaptic tagging and capture;
   - recall of the memory after 8 hours
 * `run\_smallnet3\_10s-recall\_desktop`: 
   - network of 4 excitatory and 1 inhibitory neurons;
   - one exc. neuron receives a learning stimulus;
   - the same exc. neurons receives a recall stimulus 10 seconds later
 * `run\_smallnet3\_8h-recall\_desktop`: 
   - network of 4 excitatory and 1 inhibitory neurons;
   - one exc. neuron receives a learning stimulus;
   - the same exc. neurons receives a recall stimulus 8 hours later

Besides these scripts, there are also scripts with the filename suffix `_gwdg-medium_cpu` instead of `_desktop`. These scripts are intended to run simulations on a SLURM compute cluster (as operated by [GWDG](https://gwdg.de/)).
### Tests
Integration tests are defined in `test_arborNetworkConsolidation.py` and can be run via the bash script file `run\_tests`.

### Dependencies
The code has been tested with Arbor development version v0.8.2-dev, commit [d1139b7](https://github.com/arbor-sim/arbor/commit/d1139b700db0ed640e216f76564278f14fd83dca).
To install this Arbor version from the source code, you can run the following:
```
git clone --recursive https://github.com/arbor-sim/arbor/ arbor_source_repo
mkdir arbor_source_repo/build && cd arbor_source_repo/build
git checkout -b arbor_v0.8.2-dev-d1139b7 d1139b700db0ed640e216f76564278f14fd83dca
cmake -DARB_WITH_PYTHON=ON -DARB_USE_BUNDLED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$(readlink -f ~/arbor_v0.8.2-dev-d1139b7) -DPYTHON_EXECUTABLE:FILEPATH=`which python3` -S .. -B .
#make tests && ./bin/unit # optionally: testing
make install
```
Subsequently, you can use the the script `set\_arbor\_env\_dev` to set the environment variables accordingly. This script is run automatically by the build and run scripts provided here and needs to be adapted if you use a different Arbor installation.

Build the custom catalogue of mechanisms by running `build\_catalogue`. 

Further package dependencies:
 * `matplotlib`
 * `numpy`
 * `pytest`
 * `pytest-cov`
 * `coverage`

Install for example by `python3 -m pip install <name-of-package>`.
