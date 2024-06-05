#importing packages
import torch
import numpy as np
import scipy

#importing functions from packages
from torch import tensor
from math import ceil

#importing helper functions from file
from helperFuncs import *

def SRK2(t0, tf, U0, h, H):
    '''
    DESC: Implements complex ODE solver for pyTorch or numpy matrices with Symplectic Runge-Kutta second order method. \n
    
    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - H: Time dependent Hamiltonian \n

    OUTPUT: pyTorch Tensor or numpy vector with ODE solved over time intervals (note gate may not be unitary) \n

    AUTHOR: Bora Basyildiz
    '''
    n = ceil((tf-t0)/h)
    U = U0
    t = t0
    if isinstance(U0,torch.Tensor):
        I = tensor(np.eye(len(U)))
    else:
        I = np.eye(len(U))

    for i in range(n):
        if isinstance(U0,torch.Tensor):
            k1 = torch.linalg.solve(I + 1j*h*0.25*H(t + 0.25*h), -1j*H(t+0.25*h) @ U)
            k2 = torch.linalg.solve(I + 1j*h*0.25*H(t + 0.75*h), -1j*H(t+ 0.75*h) @ U - 1j*0.5*h*H(t+0.75*h) @ k1)
        else:
            k1 = scipy.linalg.solve(I + 1j*h*0.25*H(t + 0.25*h), -1j*H(t+0.25*h) @ U)
            k2 = scipy.linalg.solve(I + 1j*h*0.25*H(t + 0.75*h), -1j*H(t+ 0.75*h) @ U - 1j*0.5*h*H(t+0.75*h) @ k1)

        U = U + 0.5*h*(k1 + k2)
        t = t + h
    return U

def RKN4(t0, tf, U0, h, dUdt, H):
    '''
    DESC: Implements complex ODE solver for pyTorch/numpy matrices with Runge-Kutta fourth order method with norm normalization. \n
    
    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - dUdt: Schrödinger equation 
        - H: Time dependent Hamiltonian \n

    OUTPUT: pyTorch/numpy matrix with ODE solved over time intervals (note gate may not be unitary) \n

    AUTHOR: Bora Basyildiz & Will Beason
    '''
    n = ceil((tf-t0)/h)
    U = U0
    t = t0
    for i in range(n):
        k1 = normU(dUdt(t + h/2,U, H))
        k2 = normU(dUdt(t + h/2, U + h/2*k1, H))
        k3 = normU(dUdt(t + h/2, U + h/2*k2, H))
        k4 = normU(dUdt(t + h, U + h*k2, H))

        U = U + h/6 * (k1+2*k2 + 2*k3 + k4)
        t = t + h
    return U

def RKN2(t0, tf, U0, h, dUdt, H):
    '''
    DESC: Implements complex ODE solver for pyTorch/numpy matrices with Runge-Kutta second order method with norm normalization. \n
    
    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - dUdt: Schrödinger equation 
        - H: Time dependent Hamiltonian \n

    OUTPUT: pyTorch/numpy matrix with ODE solved over time intervals (note gate may not be unitary) \n

    AUTHOR: Bora Basyildiz
    '''
    n = ceil((tf-t0)/h)
    U = U0
    t = t0

    for i in range(n):
        k1 = normU(dUdt(t, U, H))
        k2 = normU(dUdt(t + 0.5*h, U + 0.5*h*k1, H))

        U = U + h*k2
        t = t + h
    return U

def SV2(t0, tf, U0, h, H):
    '''
    DESC:  Implements complex ODE solver for pyTorch/numpy matrices with Störmer-Verlet symplectic ODE Solver (Second Order). \n

    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - H: Time dependent Hamiltonian \n

    OUTPUTS: pyTorch/numpy matrix with ODE solved over time intervals (note gate may not be unitary) \n

    AUTHOR: Bora Basyildiz
    '''
    n = ceil((tf-t0)/h)
    tn = t0
    if isinstance(U0,torch.Tensor):
        Un = torch.real(U0)
        Vn = -torch.imag(U0)
        I = tensor(np.eye(len(Un)))
    else:
        Un = np.real(U0)
        Vn = -np.imag(U0)
        I = np.eye(len(Un))

    for i in range(n):
        # Substep and subcall generating
        U1 = Un
        if isinstance(U0,torch.Tensor):
            l1 = torch.linalg.solve(I - (h/2) * S(tn + h/2, H), K(tn + h/2, H) @ U1 + S(tn + h/2, H) @ Vn)
        else:
            l1 = scipy.linalg.solve(I - (h/2) * S(tn + h/2, H), K(tn + h/2, H) @ U1 + S(tn + h/2, H) @ Vn)
        V1 = Vn + (h/2) * l1
        V2 = V1
        k1 = fu(U1, V1, tn, H)
        if isinstance(U0,torch.Tensor):
            k2 = torch.linalg.solve(I - (h/2) * S(tn + h, H), S(tn + h, H) @ (Un + (h/2) * k1) - K(tn + h, H) @ V1)
        else:
            k2 = scipy.linalg.solve(I - (h/2) * S(tn + h, H), S(tn + h, H) @ (Un + (h/2) * k1) - K(tn + h, H) @ V1)
        U2 = Un + (h/2) * (k1 + k2)
        l2 = fv(U2, V2, tn + h/2, H)

        # Time-step Evolution
        Un = Un + (h/2) * (k1 + k2)
        Vn = Vn + (h/2) * (l1 + l2)
        tn = tn + h
    return Un - 1j*Vn
    

def RK4(t0, tf, U0, h, dUdt, H):
    '''
    DESC: Implements complex ODE solver for pyTorch/numpy matrices with Runge-Kutta fourth order method. \n

    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - dUdt: Schrödinger equation 
        - H: Time dependent Hamiltonian \n

    OUTPUT: pyTorch/numpy matrix with ODE solved over time intervals (note gate may not be unitary) \n

    AUTHOR: Bora Basyildiz & Prateek Bhindwar
    '''
    # number of iterations based on dt(h)
    n = ceil((tf-t0)/h)
    U = U0
    for i in range(n):
        #RK4 method
        k1 = h * dUdt(t0, U, H)
        k2 = h * dUdt(t0 + 0.5 * h, U + 0.5 * k1, H)
        k3 = h * dUdt(t0 + 0.5 * h, U + 0.5 * k2, H)
        k4 = h * dUdt(t0 + h, U + k3, H)
 
        # Updating U and time-step
        U = U + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        t0 = t0 + h
    return U

def RK2(t0, tf, U0, h, dUdt, H):
    '''
    DESC: Implements complex ODE solver for pyTorch/numpy matrices with Runge-Kutta second order method. \n

    PARAMS: 
        - t0: start time 
        - tf: end time 
        - U0: Initial unitary Matrix
        - h: change in time for each step 
        - dUdt: Schrödinger equation 
        - H: Time dependent Hamiltonian \n

    OUTPUT: pyTorch/numpy matrix with ODE solved over time intervals (note gate may not be unitary) \n

    AUTHOR: Bora Basyildiz & Prateek Bhindwar
    '''
    n = ceil((tf-t0)/h)
    U = U0
    t = t0

    for i in range(n):
        k1 = dUdt(t, U, H)
        k2 = dUdt(t + 0.5*h, U + 0.5*h*k1, H)

        U = U + h*k2
        t = t + h
    return U