import numpy as np



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






    

def spinCouplingClosed(spins,magMom):
    -spins
    return