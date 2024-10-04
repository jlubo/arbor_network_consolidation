# Arbor simulation of memory consolidation in recurrent spiking neural networks based on synaptic tagging and capture

This package serves to simulate recurrent spiking neural networks, consisting of leaky integrate-and-fire neurons connected via current-based plastic synapses, with the Arbor simulator library.
The long-term plasticity model that is employed features a calcium-based early phase and a late phase that is based on synaptic tagging-and-capture mechanisms. 
The underlying model has been described in detail in [Luboeinski and Tetzlaff (2021)](https://doi.org/10.1038/s42003-021-01778-y) and has previously been implemented with a [stand-alone simulator](https://github.com/jlubo/memory-consolidation-stc). The code provided here can reproduce the previous results with the Arbor simulator library.
An implementation that employs Arbor to simulate similar networks with morphological neurons will be made available soon.

### Provided code
The main simulation code is found in `arborNetworkConsolidation.py`. The code in `plotResults.py` is used to plot the results and is called automatically after a simulation is finished (but it can also be used on its own). The file `outputUtilities.py` provides additional utility functions needed. The parameter configurations for different types of simulations are provided by means of `*.json` files. The neuron and synapse mechanisms are provided in the [Arbor NMODL format](https://docs.arbor-sim.org/en/v0.10.0/fileformat/nmodl.html) in the files `mechanisms/*.mod`.

To achieve viable runtimes even for long biological timescales, the program determines a "schedule" for each simulation (e.g., "learn - consolidate"). This causes the simulation to either run in one piece, or to be split up into different phases which are computed using different timesteps (small timesteps for detailed dynamics and substantially longer timesteps for phases of plasticity consolidation; in the latter case, the spiking and calcium dynamics are neglected, and only the late-phase dynamics and the exponential decay of the early-phase weights are computed, the validty of which has been shown [here](https://doi.org/10.53846/goediss-463)). The plasticity mechanism (`expsyn_curr_early_late_plasticity` or `expsyn_curr_early_late_plasticity_ff`) is chosen accordingly.

Different simulation protocols can easily be run using the following bash script files:
 * `run_basic_early` 
   - example of basic early-phase plasticity dynamics in a small network
 * `run_basic_late`
   - example of basic late-phase plasticity dynamics in a small network
 * `run_batch_basic_early` 
   - run batch of basic early-phase plasticity dynamics in a small network (employs `runBatchesBasic.py`)
 * `run_batch_basic_early`
   - run batch of basic late-phase plasticity dynamics in a small network (employs `runBatchesBasic.py`)
 * `run_benchmark_desktop`
   - pipeline for benchmarks of runtime and memory usage (for the latter, the script `track_allocated_memory` is used); 
   - can be used with different paradigms (`CA200`, `2N1S_basic_late`, ...);
   - script with suffix `_gpu` can be used for benchmarking on GPU systems
 * `run_defaultnet_bg_only_desktop` 
   - network of 1600 excitatory and 400 inhibitory neurons;
   - background input only
 * `run_defaultnet_10s-recall_desktop`
   - network of 1600 excitatory and 400 inhibitory neurons;
   - learning a memory represented by a cell assembly of a specified number of exc. neurons (_argument $1_);
   - recall of the memory after 10 seconds;
   - script with suffix `_gpu` can be used for running simulations on GPU systems
 * `run_defaultnet_8h-recall_desktop`
   - network of 1600 excitatory and 400 inhibitory neurons;
   - learning a memory represented by a cell assembly of a specified number of exc. neurons (_argument $1_);
   - consolidation via synaptic tagging and capture;
   - recall of the memory after 8 hours;
   - script with suffix `_gpu` can be used for running simulations on GPU systems
 * `run_smallnet3_10s-recall_desktop`: 
   - network of 4 excitatory and 1 inhibitory neurons;
   - one exc. neuron receives a learning stimulus;
   - the same exc. neurons receives a recall stimulus 10 seconds later
 * `run_smallnet3_8h-recall_desktop`: 
   - network of 4 excitatory and 1 inhibitory neurons;
   - one exc. neuron receives a learning stimulus;
   - the same exc. neurons receives a recall stimulus 8 hours later

Besides these scripts, there are also scripts with the filename suffix `_gwdg-*` (instead of `_desktop`). Those scripts are intended to run simulations on a SLURM compute cluster (as operated by [GWDG](https://gwdg.de/)).

### Tests
Integration tests are defined in `test_arborNetworkConsolidation.py` and can be run via the bash script file `run_tests`.

### Arbor installation
The code has been tested with Arbor version v0.10.0. 

#### Default version
Most conveniently, you can install this Arbor version via
```
python3 -m pip install arbor==0.10.0
```

To install this Arbor version from source code (default version, without SIMD support), you can run the following:
```
git clone --recursive https://github.com/arbor-sim/arbor/ arbor_source_repo
mkdir arbor_source_repo/build && cd arbor_source_repo/build
git checkout 6b6cc900b85fbf833fae94817b9406a0d690dc28 -b arbor_v0.10.0
cmake -DARB_WITH_PYTHON=ON -DARB_USE_BUNDLED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$(readlink -f ~/arbor_v0.9.1-dev-plastic_arbor_v2-nosimd) -DPYTHON_EXECUTABLE:FILEPATH=`which python3.10` -S .. -B .
#make tests && ./bin/unit # optionally: testing
make install
```

#### SIMD version
To install this Arbor version from source code (with SIMD support), you can run the following:
```
git clone --recursive https://github.com/arbor-sim/arbor/ arbor_source_repo
mkdir arbor_source_repo/build && cd arbor_source_repo/build
git checkout 6b6cc900b85fbf833fae94817b9406a0d690dc28 -b arbor_v0.10.0
cmake -DARB_WITH_PYTHON=ON -DARB_USE_BUNDLED_LIBS=ON -DARB_VECTORIZE=ON -DCMAKE_INSTALL_PREFIX=$(readlink -f ~/arbor_v0.9.1-dev-plastic_arbor_v2-simd) -DPYTHON_EXECUTABLE:FILEPATH=`which python3.10` -S .. -B .
#make tests && ./bin/unit # optionally: testing
make install
```
You may also take this shortcut with `pip`:
```
CMAKE_ARGS="-DARB_VECTORIZE=ON" python3 -m pip install ./arbor_source_repo
```

#### CUDA version
To install this Arbor version from source code (with CUDA GPU support), you can run the following (cudatoolkit v11.5 and gcc/g++ v10 are recommended):
```
git clone --recursive https://github.com/arbor-sim/arbor/ arbor_source_repo
mkdir arbor_source_repo/build && cd arbor_source_repo/build
git checkout 6b6cc900b85fbf833fae94817b9406a0d690dc28 -b arbor_v0.10.0
CC=gcc-10 CXX=g++-10 cmake -DARB_WITH_PYTHON=ON -DARB_USE_BUNDLED_LIBS=ON -DARB_GPU=cuda -DARB_USE_GPU_RNG=ON -DCMAKE_INSTALL_PREFIX=$(readlink -f ~/arbor_v0.9.1-dev-plastic_arbor_v2-cuda_use_gpu_rng) -DPYTHON_EXECUTABLE:FILEPATH=`which python3.10` -S .. -B .
#make tests && ./bin/unit # optionally: testing
make install
```
You may also take this shortcut with `pip`:
```
CMAKE_ARGS="-DARB_GPU=cuda" python3 -m pip install ./arbor_source_repo
```

#### Environment setting

You can use the script `set_arbor_env` (or `set_arbor_env_*`) to set the environment variables for your Arbor installation. You can then build the custom catalogue of mechanisms by running `build_catalogue` (or `build_catalogue_*`). Such a script is run automatically by any run script provided here and needs to be adapted if you use a different Arbor installation.

### Further dependencies
Further package dependencies:
 * `matplotlib`
 * `numpy`
 * `pytest`
 * `pytest-cov`
 * `coverage`

You can install them, for example, via `python3 -m pip install <name-of-package>` (if necessary, upgrade them via `python3 -m pip install -U <name-of-package>`).
