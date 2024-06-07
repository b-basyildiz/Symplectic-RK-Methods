# Symplectic-RK-Methods
## Intro
Here we will implement numerical ordinary differential equation solvers (N-ODEs). Specifically we will implement Symplectic Runge-Kutta methods and Runge-Kutta methods that solve real and imaginary equations. Also we will also have integration with the commonly used optimization package pytorch.

For an usage case of this package see [arXiv:2312.09218](https://arxiv.org/abs/2312.09218). Here symplectic runge kutta methods were used to model the time-dependent evolution of a quantum system. This quantum system is determined by the Schrödinger equation $i\hbar\frac{\partial}{\partial t}|\psi\rangle = H(t)|\psi\rangle$. Also, the solution of this equation was used in the cost function of a PyTorch parameter optimization. Note that I am the lead author of this paper. 

## Runge-Kutta Methods for Imaginary Equations
Here we will detail standard Runge-Kutta (RK) methods for solving imaginary equations. While RK methods are easily to code for real and imaginary equations, RK methods for imaginary equations are not integrated into PyTorch. We will provide methods that integrate with the standard package numpy and PyTorch. 

Traditional Runge-Kutta (RK) methods are well developed and understood and are their functions are usually coding excercises for applied mathematicians and computer scientists. The RK methods detailed here are not widely used. Symplectic RK methods are unique such that they preserve the unitary evolution of an ODE system (real or imaginary). This can be thought of conserving the probabilities in a probability vector or making sure the evolution of a vector is only rotations and does not include any scalings. 


