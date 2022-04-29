import numba as fast
from jitConsts import Consts
import numpy as np



def selectmode(periodic,xmax,ymax):
    if periodic:
        if xmax == 1:
            mode = latticeChangePeriodic0d_X
        elif ymax == 1:
            mode = latticeChangePeriodic0d_Y
        else:
            mode = latticeChangePeriodic2d
    else:
        mode = latticeChangeNonPeriodic
    return mode


@fast.njit(cache = True)
def dtSpin(S : np.ndarray, F : np.ndarray, GAMMA, ALPHA):
    prefac = -GAMMA/(1+ALPHA**2) 
    terms = np.cross(S,F)+ALPHA*np.cross(S,np.cross(S,F)) 
    return prefac*terms


@fast.njit(parallel = True, cache = True)
def latticeChangeNonPeriodic(lattice, randnums, dt, xmax, ymax, C : Consts):
    ez = np.array([0,0,1])
    tempLattice = np.zeros(shape = (ymax+1,xmax+1,3))
    res = np.empty(shape = (ymax,xmax,3))

    randomscaling = np.sqrt(2*C.ALPHA*C.KBT/(C.GAMMA*C.magMom*dt))
    tempLattice[:-1,:-1] = lattice
    for y in fast.prange(ymax):
        
        for x in range(xmax):

            anis = 2*C.d_z*np.linalg.norm(tempLattice[y,x],ord = 1)*ez
            
            spincoupling = C.J * (tempLattice[y,(x+1)] + tempLattice[y,x-1] + tempLattice[(y+1),x] + tempLattice[y-1,x]) 
            
            Fj = C.B + spincoupling/C.magMom + anis/C.magMom

            spinChange = dtSpin(lattice[y,x],Fj,C.GAMMA,C.ALPHA)
            
            res[y,x] = spinChange + randnums[y,x]*randomscaling
    return res

@fast.njit(parallel = True, cache = True)
def latticeChangePeriodic2d(lattice, randnums, dt, xmax, ymax, C : Consts):
    ez = np.array([0,0,1])
    tempLattice = lattice.copy()

    res = np.empty(shape = (ymax,xmax,3))

    randomscaling = np.sqrt(2*C.ALPHA*C.KBT/(C.GAMMA*C.magMom*dt))

    for y in fast.prange(ymax):
        
        for x in range(xmax):

            anis = 2*C.d_z*np.linalg.norm(tempLattice[y,x],ord = 1)*ez
            
            spincoupling = C.J * (tempLattice[y,(x+1)%xmax] + tempLattice[y,x-1] + tempLattice[(y+1)%ymax,x] + tempLattice[y-1,x]) 
            
            Fj = C.B + spincoupling/C.magMom + anis/C.magMom

            spinChange = dtSpin(lattice[y,x],Fj,C.GAMMA,C.ALPHA)
            
            res[y,x] = spinChange + randnums[y,x]*randomscaling
    return res

@fast.njit(parallel = True, cache = True)
def latticeChangePeriodic0d_X(lattice, randnums, dt, xmax, ymax, C : Consts):
    ez = np.array([0,0,1])
    tempLattice = lattice.copy()
    
    res = np.empty(shape = (ymax,xmax,3))

    randomscaling = np.sqrt(2*C.ALPHA*C.KBT/(C.GAMMA*C.magMom*dt))

    for y in fast.prange(ymax):
        
        for x in range(xmax):

            anis = 2*C.d_z*np.linalg.norm(tempLattice[y,x],ord = 1)*ez
            
            spincoupling = C.J * ( tempLattice[(y+1)%ymax,x] + tempLattice[y-1,x]) 
            
            Fj = C.B + spincoupling/C.magMom + anis/C.magMom

            spinChange = dtSpin(lattice[y,x],Fj,C.GAMMA,C.ALPHA)
            
            res[y,x] = spinChange + randnums[y,x]*randomscaling
    return res

@fast.njit(parallel = True, cache = True)
def latticeChangePeriodic0d_Y(lattice, randnums, dt, xmax, ymax, C : Consts):
    ez = np.array([0,0,1])
    tempLattice = np.zeros(shape = (ymax,xmax+1,3))
    tempLattice = lattice.copy()
    
    res = np.empty(shape = (ymax,xmax,3))

    randomscaling = np.sqrt(2*C.ALPHA*C.KBT/(C.GAMMA*C.magMom*dt))

    for y in fast.prange(ymax):
        
        for x in range(xmax):

            anis = 2*C.d_z*np.linalg.norm(tempLattice[y,x],ord = 1)*ez
            
            spincoupling = C.J * (tempLattice[y,(x+1)%xmax] + tempLattice[y,x-1]) 
            
            Fj = C.B + spincoupling/C.magMom + anis/C.magMom

            spinChange = dtSpin(lattice[y,x],Fj,C.GAMMA,C.ALPHA)
            
            res[y,x] = spinChange + randnums[y,x]*randomscaling
    return res

