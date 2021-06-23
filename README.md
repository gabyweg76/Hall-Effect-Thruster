# Hall-Effect-Thruster

All the source codes are available on this repository.

## Finite element method using DSMC method for a Hybrid-PIC code:

A representation of a plasma flow for an electrostatic problem. We represent our plasma as an assembly od particles circulating under an static electromagnetic constraint. For an unstructured mesh, a Direct Simulation Monte Carlo method is introduced to solve the [Poisson's equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) with an hybrid approach:
- the ions are represented as particles
- the electrons are represented as a fluid based upon the [Boltzmann's equation](https://en.wikipedia.org/wiki/Boltzmann_equation)

Two types of codes are avaible in `DSMC Hybrid-PIC`
This code inspired from the work of [Lubos Breida](https://www.particleincell.com/2015/fem-pic/). 
