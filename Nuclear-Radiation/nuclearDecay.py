import numpy as np
from numba import njit, vectorize
import matplotlib.pyplot as plt
@njit
def probDecayed(lamb,dt):
    return 1-np.exp(-dt*lamb)

@njit
def expDecayFast(N0 : int,nDecays : int = 3 , dt : float = 0.01,steps = 100,lambdas = np.array([0.0002,0.0001,0.0002])):
    """
    Simulate a decay chain of radiative atoms.

    params:
        N0      : int        -> Start amount of first decaying atom
        nDecays : int        -> The number of daughter atoms
        dt      : float      -> Time between decay checks
        steps   : int        -> Number of decay checks
        lambdas : np.ndarray -> The respective decay rates of each atom in decay succession order
    
    returns:
        result : np.ndarray  -> The time evolution of atom populations
    """    
    results = np.zeros(shape=(steps,nDecays),dtype = np.int32)
    probs = np.empty(nDecays,dtype = np.float64)
    # calculate probability of decay after a time step dt for every atom
    for i in range(nDecays):
        probs[i] = probDecayed(lambdas[i],dt)

    # Initialize and evolve population
    results[0,0] = N0
    for j in range(steps-1):
        # last population is stable:
        results[j+1,-1] = results[j,-1]

        # loop backwards through populations, to only decay the atoms that have
        # been in the population for at least a time length dt 
        for i in range(nDecays-2,-1, -1):
            # set population to be equal last step's population to ensure 
            results[j+1,i] = results[j,i]  

            #loop through all atoms and check if they decayed, if so, update populations
            for atom in range(results[j,i]):
                if np.random.random() < probs[i] :
                    results[j+1,i] -= 1 
                    results[j+1,i+1] += 1
            
    return results

def q2():
    steps = 15000
    dt = 0.001
    t = np.arange(steps)
    lambdas = np.array([0.3,0.02,0.0002])
    n = 1000
    M = 1000
    a = np.empty(steps,dtype = np.float64)
    for i in range(n):
        results = expDecayFast(M,2,dt=dt,steps = steps,lambdas=lambdas)
        for i,N in enumerate(results.T):
            #plt.plot(t*0.01,N, label = i, linestyle = "--",linewidth = 1)
            if i == 0:
                a+=N/n
    plt.plot(t*dt,M*np.exp(-0.3*t*dt))
    plt.plot(t*dt,a, label = i, linestyle = "-.",linewidth = 1)
    plt.plot(t*dt,M*np.exp(-0.3*t*dt)-a)
    #plt.hlines(10000*lambdas[0]/lambdas[1],0,steps)
    plt.legend()
    plt.show()
