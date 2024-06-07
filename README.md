# Symplectic-RK-Methods
## Intro
Here we will implement numerical ordinary differential equation solvers (N-ODEs). Specifically we will implement Symplectic Runge-Kutta methods and Runge-Kutta methods that solve real and imaginary equations. Also we will also have integration with the commonly used optimization package pytorch.

For an usage case of this package see [arXiv:2312.09218](https://arxiv.org/abs/2312.09218). Here symplectic runge kutta methods were used to model the time-dependent evolution of a quantum system. This quantum system is determined by the Schrödinger equation $i\hbar\frac{\partial}{\partial t}|\psi\rangle = H(t)|\psi\rangle$. Also, the solution of this equation was used in the cost function of a PyTorch parameter optimization. Note that I am the lead author of this paper. 

## Runge-Kutta Methods for Imaginary Equations
Here we will detail standard Runge-Kutta (RK) methods for solving imaginary equations. While RK methods are easily to code for real and imaginary equations, RK methods for imaginary equations are not integrated into PyTorch. We will provide methods that integrate with the standard package numpy and PyTorch. 

Traditional Runge-Kutta (RK) methods are well developed and understood and are their functions are usually coding excercises for applied mathematicians and computer scientists. The RK methods detailed here are not widely used. Symplectic RK methods are unique such that they preserve the unitary evolution of an ODE system (real or imaginary). This can be thought of conserving the probabilities in a probability vector or making sure the evolution of a vector is only rotations and does not include any scalings. Now these methods can be found Kang and Mengzhao [1]. We can have an order (essentially the precision and complexity) of the Symplectic RK (SRK) methods, but we will choose SRK order 2 for algorithmic simplcity. If more precision is required, decreasing the respective step size $h$ is usually the best course. Here we will detail the steps fo SRK-2 for $n$ steps for solving a schödinger equation for the evolution of $H(t)$ such that $U$ is our total evolution for total time $T$. 

$$
\begin{align*}
    U_{n+1} &= U_n + \frac{h}{2}[k_1 + k_2]\\
    k_1 &= f(t_n + h/2, U_n + \frac{h}{4}k_1)\\
    k_2 &= f(t_n + 3h/4, U_n + h[\frac{k_1}{2} + \frac{k_2}{4}])\\
    f(t,U) &= -iH(t)U\\
    t_{n+1} &= t_n + h.
\end{align*}
$$

Note that all symplectic methods are implicit, and thus we must solve for $k_1, k_2$ such that 

$$
\begin{align*}
    k_1 &= \frac{-iH(t_n + h/4) U_n}{I + \frac{ih}{4}H(t_n + h/4)}\\
    k_2 &= \frac{-iH(t_n + 3h/4)U_n - \frac{ih}{2}H(t_n + 3h/4)k_1}{I + \frac{ih}{4}H(t_n + 3h/4)}.
\end{align*}
$$

1. Feng, Kang and Mengzhao Qin. “Symplectic Geometric Algorithms for Hamiltonian Systems.” (2010).
