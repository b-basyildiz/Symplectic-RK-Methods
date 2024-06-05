import torch
import numpy as np

from torch import trace, sqrt, matmul

def dUdt(t, U, H):
    '''
    DESC: Schrödinger equation in ODE form for unitaries. Necessary for RK4 method. \n

    PARAMS:
        - t: time the ODE is being evaluated
        - U: Unitary being evolved. 
        - H: Time dependent Hamiltonian \n

    OUTPUT: Matrix to be used for RK4 method \n

    AUTHOR: Bora Basyildiz
    '''
    if isinstance(U,torch.Tensor):
        return -1j*matmul(H(t),U)
    else:
        return -1j*np.matmul(H(t),U)

def S(t, H):
     if isinstance(H,torch.Tensor):
        return torch.imag(H(t))
     else:
        return np.imag(H(t))

def K(t, H):
    if isinstance(H,torch.Tensor):
        return torch.real(H(t))
    else:
        return np.real(H(t))
    
def fu(U, V, t, H):
    return S(t, H) @ U - K(t, H) @ V 

def fv(U, V, t, H):
    return K(t, H) @ U + S(t, H) @ V

def normU(U):
    '''
    DESC: Normalizes Unitary Matrix \n
    
    PARAMS:
        - U: Unitary Matrix (pyTorch tensor) \n
    
    OUTPUT: Normalized Unitary Matrix \n

    AUTHOR: Bora Basyildiz
    '''
    if isinstance(U,torch.Tensor):
        norm = sqrt(trace(matmul(U.conj().T,U))/len(U))
    else: 
        norm = np.sqrt(np.trace(np.matmul(U.conj().T,U))/len(U))
    return U/norm

def normPrint(U):
    if isinstance(U,torch.Tensor):
        print(sqrt(trace(matmul(U.conj().T,U))/len(U)))
    else:
        print(np.sqrt(np.trace(np.matmul(U.conj().T,U))/len(U)))