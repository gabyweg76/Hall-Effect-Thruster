# Hall-Effect-Thruster

All the source codes are available on this repository.

## Finite element method using DSMC method for a Hybrid-PIC code:

A representation of a plasma flow for an electrostatic problem. We represent our plasma as an assembly od particles circulating under an static electromagnetic constraint. For an unstructured mesh, a Direct Simulation Monte Carlo method is introduced to solve the [Poisson's equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) with an hybrid approach:
- the ions are represented as particles
- the electrons are represented as a fluid based upon the [Boltzmann's equation](https://en.wikipedia.org/wiki/Boltzmann_equation)

Two types of codes are avaible in `DSMC Hybrid-PIC`/
- `HET`, a simple one file code that can be run on 1 CPU.
- `MT`, a multithreading approach to compute the number density of ions faster.

To compile the code, we suggest to enter into the shell the following instructions:
```
$ mkdir results
$ g++ -std=c++11 -O2 Het.cpp -o Het 
$ ./HET
```
In the case of a multithreading approach, a shell script has been added in `DSMC Hybrid-PIC`, called `MT.sh`.
This code inspired from the work of [Lubos Breida](https://www.particleincell.com/2015/fem-pic/). 
