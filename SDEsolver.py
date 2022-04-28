import numpy as np
import numba as fast
import matplotlib.pyplot as plt
import seaborn as sns
import logging as log
from matplotlib import animation

log.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=log.INFO,
        datefmt='%H:%M:%S'
    )

# ---- Globals ----
ALPHA = 0.1 #   0 < ALPHA < 1
GAMMA = 1 #   0 < GAMMA 
J = 1     # 
KBT = J*0.01 
dt = 0.001 #picoseconds? 
# -----------------

def spinlattice(N, M,periodic = True):
    """
    Generate/initialize NxM spinlattice 

    returns:
        spinlattice : np.ndarray(shape = (N, N, 3))
    """
    initial = np.zeros(shape = (N+(not periodic),M + (not periodic),3))


    if not periodic:            
        initial[:-1,:-1] = np.array([-1,0,0])
    return initial


@fast.njit(cache = True)
def dtSpin(S : np.ndarray, F : np.ndarray):
    prefac = -GAMMA/(1+ALPHA**2) 
    terms = np.cross(S,F)+ALPHA*np.cross(S,np.cross(S,F)) 
    return prefac*terms

@fast.njit()
def effectiveFlatticeNonPeriodic(lattice,magMom,d_z, B: np.ndarray, J : float, dt : float):
    yMax = len(lattice[0])
    xMax = len(lattice[:,0])
    ez = np.array([1,0,0])
    nextLattice = lattice.copy()
    for y in fast.prange(yMax-1):
        for x in range(xMax-1):
            
            anis = 2*d_z*np.linalg.norm(lattice[y,x],ord = 2)*ez
            
            spincoupling = J * (lattice[y,x+1] + lattice[y,x-1] + lattice[y+1,x] + lattice[y-1,x]) 
            
            Fj = B + spincoupling/magMom + anis/magMom
            
            spinForce = dtSpin(lattice[y,x],Fj)

            for i in range(3):
                nextLattice[y,x,i] = lattice[y,x,i] + dt*spinForce[i]

            nextLattice[y,x] = nextLattice[y,x]/np.linalg.norm(nextLattice[y,x])
    
    return nextLattice

@fast.njit(cache = True)
def findNextLatticeEuler(lattice, randnums, dt, xmax, ymax, d_z,magMom,B):
    ez = np.array([1,0,0])
    fnext = lattice.copy()  
    randomscaling = np.sqrt(2*ALPHA*KBT/(GAMMA*magMom*dt))

    for y in fast.prange(ymax):
        for x in range(xmax):
            anis = 2*d_z*np.linalg.norm(lattice[y,x],ord = 2)*ez
            
            spincoupling = J * (lattice[y,x+1] + lattice[y,x-1] + lattice[y+1,x] + lattice[y-1,x]) 

            Fj = B + spincoupling/magMom + anis/magMom

            spinForce = dtSpin(lattice[y,x],Fj)

            for i in range(3):
                fnext[y,x,i] = lattice[y,x,i] + dt*spinForce[i] + dt*randnums[y,x,i]*randomscaling

            fnext[y,x] = fnext[y,x]/np.linalg.norm(fnext[y,x])
    return fnext



#@fast.njit(cache = True)
def HeunsMethod3dLattice(lattice : np.ndarray, dt : float, steps :int, d_z : float, magMom : float, B : np.ndarray,periodic = False):
    adjustForPeriodic = (not periodic)
    yMax = len(lattice[0]) 
    xMax = len(lattice[:,0]) 
    lattices = np.empty(shape =(steps+1,yMax,xMax,3), dtype = np.float64)
    lattices[0] = lattice

    yMax = len(lattice[0]) -adjustForPeriodic
    xMax = len(lattice[:,0]) -adjustForPeriodic

    fnext = lattice.copy()
    ez = np.array([1,0,0]) 
    adjustForPeriodic = (not periodic)
    mean = np.array([0,0,0])
    covs = np.eye(3)
    for t in range(steps):
        randnums = np.random.multivariate_normal(mean,covs,size = (yMax,xMax))
        print(randnums)
        fnext  = findNextLatticeEuler(lattices[t],randnums,dt,xMax,yMax,d_z,magMom,B)
        lattices[t+1] = 1/2 *(fnext + findNextLatticeEuler(fnext,randnums,dt,xMax,yMax,d_z,magMom,B))

    return lattices





N, M = 20,20
s = spinlattice(N,M,0)
s[0,0] = np.array([0,0,1])
plottingPlane = (0,1,2)  # (0, 1, 2) corresponds to the coordinates (x, y, z) respectively


log.info("Running...")
A = HeunsMethod3dLattice(s,0.003,100,0.1,5.788*10**-2,0*np.array([0,0,1]))
log.info("Heuns done!")
fig, ax = plt.subplots(1)

X = np.arange(N)
Y = np.arange(M)
X,Y = np.meshgrid(X,Y)


Q = ax.quiver(X,Y,A[0,:-1,:-1,plottingPlane[0]],A[0,:-1,:-1,plottingPlane[1]], A[0,:-1,:-1,plottingPlane[2]])

def update_quiver(i,Q,A):
    """
    updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    U = A[i,:-1,:-1,plottingPlane[0]]
    V = A[i,:-1,:-1,plottingPlane[1]]
    C =  A[i,:-1,:-1,plottingPlane[2]]
    
    Q.set_UVC(U,V,C)
    return Q,
log.info("Creating animation")
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, A),
                               interval=30, blit=False, repeat = True)
plt.show()
"""

palette = sns.color_palette("viridis",30)

N, M = 20,20
A = spinlattice(N,M,0)
A[0,0] = np.array([0,0,1.0])

a = []

fig, ax = plt.subplots(1)

X = np.arange(N)
Y = np.arange(M)
X,Y = np.meshgrid(X,Y)


Q = ax.quiver(X,Y,A[:-1,:-1,1],A[:-1,:-1,2])

plt.xlim([-1,N+1])
plt.ylim([-1,M+1])

for i in range(10000):
    a.append(A)
    A = effectiveFlatticeNonPeriodic(A,0.05,0.1,np.array([0.0,0,0]),1,0.0003)
    
    for k in range(5):
        for j in range(5):
            plt.quiver(j,k,A[j,k,1],A[j,k,2], color = palette[i])



def update_quiver(i,Q,a):
    updates the horizontal and vertical vector components by a
    fixed increment on each frame
    
    U = a[i*5][:-1,:-1,1]
    V = a[i*5][:-1,:-1,2]
    
    Q.set_UVC(U,V)
    return Q,


anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, a),
                               interval=25, blit=False, repeat = True)
plt.show()"""
