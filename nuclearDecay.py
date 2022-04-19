import numpy as np
from numba import njit, vectorize
import matplotlib.pyplot as plt
@njit
def probDecayed(lamb,dt):
    return 1-np.exp(-dt*lamb)

@njit
def expDecayFast(N0,nDecays = 3,dt = 0.01,steps = 100,lambdas = np.array([0.0002,0.0001,0.0002])):
    results = np.zeros(shape=(steps,nDecays),dtype = np.int32)
    probs = np.empty(nDecays,dtype = np.float64)
    for i in range(nDecays):
        probs[i] = probDecayed(lambdas[i],dt)
    results[0,0] = N0
    for j in range(steps-1):
        results[j+1,-1] = results[j,-1]
        for i in range(nDecays-2,-1, -1):
            results[j+1,i] = results[j,i]  
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
    M = 10000
    a = np.empty(steps,dtype = np.float64)
    for i in range(n):
        results = expDecayFast(M,2,dt=dt,steps = steps,lambdas=lambdas)
        for i,N in enumerate(results.T):
            #plt.plot(t*0.01,N, label = i, linestyle = "--",linewidth = 1)
            if i == 0:
                a+=N/n
    plt.plot(t*dt,M*np.exp(-0.3*t*dt))
    plt.plot(t*dt,a, label = i, linestyle = "-.",linewidth = 1)
    plt.plot(t*dt,M*np.exp(-0.3*t*dt)[::10]-a)
    #plt.hlines(10000*lambdas[0]/lambdas[1],0,steps)
    plt.legend()
    plt.show()
q2()