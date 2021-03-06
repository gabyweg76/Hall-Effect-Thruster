# Hall-Effect-Thruster

All the source codes are available on this repository.

## Finite element method using DSMC method for a Hybrid-PIC code

A representation of a plasma flow for an electrostatic problem. We represent our plasma as an assembly od particles circulating under an static electromagnetic constraint. For an unstructured mesh, a Direct Simulation Monte Carlo method is introduced to solve the [Poisson's equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) with an hybrid approach:
- the ions are represented as particles
- the electrons are represented as a fluid based upon the [Boltzmann's equation](https://en.wikipedia.org/wiki/Boltzmann_equation)

Two types of codes are avaible in `DSMC Hybrid-PIC`
- `HET`, a simple one file code that can be run on 1 CPU.
- `MT`, a multithreading approach to compute the number density of ions faster.

To compile the code, we suggest to enter into the shell the following instructions:
```
$ mkdir results
$ g++ -std=c++11 -O2 Het.cpp -o Het 
$ ./HET
```
In the case of a multithreading approach, a shell script has been added in `DSMC Hybrid-PIC`, called `MT.sh`.
This code got inspired from the work of [Lubos Breida](https://www.particleincell.com/2015/fem-pic/). 

## Post-Treatment Analysis using Multiclass Logistic Regression

A softmax regression code was used to predict the magnetic confinement of ions throughout the simulation.

Two types of code are available in `Machine Learning`:

- `Simulation 1` uses a simple code of softmax regression for small dataset. 
- `Simulation 2` uses [Distributed Data Parallel ](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) from Pytorch to speed up the training process in case of bigger datasets.

All used datasets are available in the repository (`*.h5` files). 

These codes got inspired from the work of [Sebastian Raschka](https://github.com/rasbt/deeplearning-models). 

## Codes compiled with Compute Canada Clusters

All the following code have been executed from Béluga, a cluster belonging to Compute Canada. For more information about the functionning of the latter, please visit the following guide from [Compute Canada](https://docs.computecanada.ca/wiki/Running_jobs).
