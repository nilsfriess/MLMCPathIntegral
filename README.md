# Computing Path Integrals using (Multilevel) MCMC 
This repository contains a C++20 library to compute expected values for quantum mechanical systems using Feynman's path integral formalism.

## Prerequisites
To compile the examples you need:
+ A C++20 compiler (e.g., clang++ 15 or higher)
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
$ ./build/harmonic_oscillator "./apps/harmonic_oscillator.json"
```
The file `./apps/harmonic_oscillator.json` contains the parameters for the MCMC sampler.
