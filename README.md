# Computing Path Integrals using (Multilevel) MCMC 
This repository contains a C++20 library to compute expected values for quantum mechanical systems using Feynman's path integral formalism.

## Prerequisites
To compile the examples you need:
+ A C++20 compiler (The examples were tested with clang++-15 and g++-13)
+ CMake
+ A BLAS installation
+ A LAPACK installation

To install the last two on Ubuntu, you can run
```
# apt install libblas-dev liblapack-dev
```

## Running the examples
To configure the project, run
```
$ git clone --recursive https://github.com/nilsfriess/MLMCPathIntegral.git
$ cd MLMCPathIntegral
$ cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++
```
Replace the compiler `clang++` with the compiler installed on your system (e.g., `clang++-15`). Now compile the examples with
```
$ cmake --build build
```
To run, e.g., the harmonic oscillator example, run
```
$ ./build/examples/harmonic_oscillator "./examples/harmonic_oscillator.json"
```
The file `./examples/harmonic_oscillator.json` contains the parameters for the MCMC sampler.

## Acknowledgements
The single level idea is explained in [1]. The multilevel approach is from [2]; the implementation here is inspired by [this repository](https://github.com/eikehmueller/mlmcpathintegral).


> [1] Creutz, M., and B. Freedman. _A statistical approach to quantum mechanics._ Annals of Physics 132.2 (1981): 427-462. 
>
> [2] Jansen, Karl, Eike H. Müller, and Robert Scheichl. _Multilevel Monte Carlo algorithm for quantum mechanics on a lattice._ Physical Review D 102.11 (2020): 114512.
