# physics_modelling

A script for fitting a Monte Carlo simulation of the temperature of a single atom in an optical tweezer trap to measured data.

## About

This script will fit measured release-and-recapture data with a Monte Carlo thermodynamic simulation and output an estimated atom temperature and uncertainty.
It will also produce a plot of the fit.

For more details on the physics behind the code, see my [PhD Thesis](http://etheses.dur.ac.uk/14468/1/RVB_Thesis.pdf?DDD25+) or [associated paper](https://iopscience.iop.org/article/10.1088/1367-2630/ac0000).

## Contents

- [monte_carlo_atom_temperature.py](https://github.com/rvbrooks/physics_modelling/blob/main/monte_carlo_atom_temperature.py): the script that does the fitting
- plt_styles: some nice matplotlib styles for nice plots.
- monte_carlo_results: the output of the fitting
- data: a data file from a measurement performed using a Rubidium atom in an optical tweezer that is fitted by the script.


## Result:

Learn how I generated this plot: the wiggly line is the best fit of a Monte Carlo simulation to the data points, which are from real measurements of a single atom in a laser trap!

![alt text](https://github.com/rvbrooks/physics_modelling/blob/main/monte_carlo_result.png)

### <span style="color:orange">Find more of my work [here on GitHub](https://github.com/rvbrooks)!</span>

