import numpy as np


# ---- Globals ----
ALPHA = 0 #   0 < ALPHA < 1
GAMMA = 1 #   0 < GAMMA 
J = 1     #   
# -----------------

def spinlattice(N, M,periodic = True):
    """
    Generate/initialize NxM spinlattice 

    returns:
        spinlattice : np.ndarray(shape = (N, N, 3))
    """
    initial = np.zeros(shape = (N+(not periodic),M + (not periodic),3))


    if not periodic:
        initial[:-1,:-1] = np.ones(3)/np.sqrt(3)
    return initial
print(spinlattice(2,2,False))


def dtSpin(S : np.ndarray, F : np.ndarray):
    prefac = -GAMMA/(1+ALPHA**2) 
    terms = np.cross(S,F)+ALPHA*np.cross(S,np.cross(S,F)) 
    return prefac*terms

def Feffective(j,spins):
    


    return





    

def spinCouplingClosed(spins,magMom):
    -spins
    return