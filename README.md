# Description

Final project for the class on Stochastic Numerics in RWTH Aachen University,
Spring 2019 semester.

The project was developed by Alexander Glushko and Dmitry I. Kabanov.

Git repository for the project is
<https://github.com/dmitry-kabanov/stoch-num-2019-final-project>.

# Purpose of the project

The project goal is to apply neural networks to the problem of parameter
estimation in partial differential equations (PDEs).

Typically such problems are solved via nonlinear optimization such that on each
optimization step the PDE is solved using numerical methods.
Here, we replace solving the PDE with neural network.
To make sure that the neural network is close to the PDE solution, we enforce
the network to satisfy the PDE by adding it as a constraint in the optimization
problem.
