import numba as fast
from jitConsts import Consts
import numpy as np
import logging as log

def selectmode(periodic,xmax,ymax):
    """
    Selects and returns the correct evolution function based on the 
    inputs. These are optimized for specific boundary conditions.

    All functions:

    Perform an Euler - type step to calculate the next spin state. 
    They all have the same params and general function:

    params:
        lattice  : np.ndarray -> The current state to be evolved
        randnums : np.ndarray -> lattice of generated random numbers so that numba doesnt have to do this also
        dt       : float      -> time step size
        ymax     : int        -> length of y axis
        xmax     : int        -> length of x axis
        C        : Consts     -> jit-friendly class for passing the conditions onto the final function
    
    returns 
        res      : np.ndarray -> the resulting evolved state using an Euler step
    """

    if periodic:
        if xmax == 1:
            log.info("latticeChangePeriodic0d_X selected")
            mode = latticeChangePeriodic0d_X
        elif ymax == 1:
            log.info("latticeChangePeriodic0d_Y selected")
            mode = latticeChangePeriodic0d_Y
        else:
            log.info("latticeChangePeriodic2d selected")
            mode = latticeChangePeriodic2d
    else:
        log.info("latticeChangeNonPeriodic selected")
        mode = latticeChangeNonPeriodic
    return mode


@fast.njit(cache = True)
def dtSpin(S : np.ndarray, F : np.ndarray, GAMMA, ALPHA):
    prefac = -GAMMA/(1+ALPHA**2) 
    terms = np.cross(S,F)+ALPHA*np.cross(S,np.cross(S,F)) 
    return prefac*terms


@fast.njit(parallel = True, cache = True)
def latticeChangeNonPeriodic(lattice:np.ndarray, randnums:np.ndarray, dt:float, xmax:int, ymax:int, C : Consts) -> np.ndarray:
    """
    See selectmode()
    Only trickery here is the padding. To simulate zero-boundary, the lattice is padded with 0 total-spin states
    This ensures that the indices x =  xmax+1 = -1, y =  ymax+1 = -1 all point to zero-spin (0),
    generating no coupling forcing. Example:

      <- SSSSSSSSS0 <-
      <- SSSSSSSSS0 <-
         0000000000 
    Same for y dimension.

    """
    ez = np.array([0,0,1])
    
    tempLattice = np.zeros(shape = (ymax+1,xmax+1,3))
    tempLattice[:-1,:-1] = lattice

    res = np.empty(shape = (ymax,xmax,3))

    randomscaling = np.sqrt(2*C.ALPHA*C.KBT/(C.GAMMA*C.magMom*dt))
    
    for y in fast.prange(ymax):
        
        for x in range(xmax):

            anis = 2*C.d_z*tempLattice[y,0]*ez
            
            spincoupling = C.J * (tempLattice[y,(x+1)] + tempLattice[y,x-1] + tempLattice[(y+1),x] + tempLattice[y-1,x]) 
            
            Fj = C.B + spincoupling/C.magMom + anis/C.magMom + randnums[y,x]*randomscaling

            spinChange = dtSpin(lattice[y,x],Fj,C.GAMMA,C.ALPHA)
            
            res[y,x] = spinChange 

    return res

@fast.njit(parallel = True, cache = True)
def latticeChangePeriodic2d(lattice:np.ndarray, randnums:np.ndarray, dt:float, xmax:int, ymax:int, C : Consts) -> np.ndarray:
    ez = np.array([0,0,1])

    res = np.empty(shape = (ymax,xmax,3))

    randomscaling = np.sqrt(2*C.ALPHA*C.KBT/(C.GAMMA*C.magMom*dt))

    for y in fast.prange(ymax):
        
        for x in fast.prange(xmax):

            anis = 2*C.d_z*lattice[y,0]*ez
            
            spincoupling = C.J * (lattice[y,(x+1)%xmax] + lattice[y,x-1] \
                                + lattice[(y+1)%ymax,x] + lattice[y-1,x]) 
            
            Fj = C.B + spincoupling/C.magMom + anis/C.magMom + randnums[y,x]*randomscaling

            spinChange = dtSpin(lattice[y,x],Fj,C.GAMMA,C.ALPHA)
            
            res[y,x] = spinChange 

    return res

@fast.njit(parallel = True, cache = True)
def latticeChangePeriodic0d_X(lattice:np.ndarray, randnums:np.ndarray, dt:float, xmax:int, ymax:int, C : Consts) -> np.ndarray:
    ez = np.array([0,0,1])
    
    res = np.empty(shape = (ymax,xmax,3))

    randomscaling = np.sqrt(2*C.ALPHA*C.KBT/(C.GAMMA*C.magMom*dt))

    for y in fast.prange(ymax):
        
        anis = 2*C.d_z*lattice[y,0]*ez
            
        spincoupling = C.J * (lattice[(y+1)%ymax,0] + lattice[y-1,0]) 
            
        Fj = C.B + spincoupling/C.magMom + anis/C.magMom + randnums[y,0]*randomscaling

        spinChange = dtSpin(lattice[y,0],Fj,C.GAMMA,C.ALPHA)
            
        res[y,0] = spinChange 

    return res

@fast.njit(parallel = True, cache = True)
def latticeChangePeriodic0d_Y(lattice:np.ndarray, randnums:np.ndarray, dt:float, xmax:int, ymax:int, C : Consts) -> np.ndarray:
    ez = np.array([0,0,1])
    
    res = np.empty(shape = (ymax,xmax,3))

    randomscaling = np.sqrt(2*C.ALPHA*C.KBT/(C.GAMMA*C.magMom*dt))

    for x in range(xmax):

        anis = 2*C.d_z*lattice[0,x]*ez
        
        spincoupling = C.J * (lattice[0,(x+1)%xmax] + lattice[0,x-1]) 
        
        Fj = C.B + spincoupling/C.magMom + anis/C.magMom + randnums[0,x]*randomscaling
        
        spinChange = dtSpin(lattice[0,x],Fj, C.GAMMA, C.ALPHA)
        
        res[0,x] = spinChange 

    return res
