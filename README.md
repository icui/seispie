# Seispie
A GPU based full waveform inversion package

# Prerequisites

* [Numba](https://numba.pydata.org)
* [Obspy](https://obspy.org)
* [CUDA](https://developer.nvidia.com/cuda-zone)
* [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (optional, required for TigerGPU)

##### Usage
Running examples on local machine:
````
sh run_example.sh
````

Running examples on TigerGPU:
````
sh submit_example.sh
````

Run custom projects:
1. Add seispie to PATH
````
export PATH=$PATH:/path/to/seispie/scripts
export PYTHONPATH=$PYTHONPATH:/path/to/seispie
````
2. Enter project directory and run
````
cd /path/to/project
sprun
````