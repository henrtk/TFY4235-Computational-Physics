import numpy as np
import numba as fast

def HeunsMethod3dLattice(f,lattice,dt,steps):
    ts = np.linspace(0,dt*(steps),steps+1)
    ys = np.empty(shape =(steps+1,3), dtype = np.float64)
    ys[0] = lattice
    yMax = len(lattice[0])
    xMax = len(lattice[:,0])
    # intitialize a dummy f
    fNextt = lattice*0

    def eulerfnext(f,latt,t,randnum): 
        for y in fast.prange(yMax):
            for x in range(xMax):
                for i in range(3):
                    latticenext = ys[y,x,i] + dt*f(ys[y,x,i])
        return latticenext


    for t in ts:
        

        for y in fast.prange(yMax):
            for x in range(xMax):
                for i in range(3):
                    yeuler = ys[y,x,i] + dt*f(ys[y,x,i])
                    ys[x+1,i] = ys[x,i] + dt/2 *(f(ys[x,i],t) + f(yeuler))
    return ys    
