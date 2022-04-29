from locale import normalize
import numpy as np
import numba as fast
from numba import float64, int32
import matplotlib.pyplot as plt
import logging as log
from matplotlib import animation
import scipy

log.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=log.INFO,
        datefmt='%H:%M:%S'
    )

ConstTypes = [("ALPHA" , float64),("GAMMA" , float64),("J",float64),
("KBT", float64),("B",float64[:]),("d_z" , float64),("magMom",float64)]

@fast.experimental.jitclass(ConstTypes)
class Consts(object):
    """
    Compact Numba-readable Costs object that carries the simulation constants.
    """
    def __init__(self,ALPHA,GAMMA,J,KBT,B,d_z,magMom):
        self.ALPHA = ALPHA      # 0 < ALPHA < 1
        self.GAMMA = GAMMA      # 0 < GAMMA 
        self.J = J
        self.KBT = KBT
        self.B = B
        self.d_z = d_z
        self.magMom = magMom


# ---- Globals ----
ALPHA = 0.1#   0 < ALPHA < 1
GAMMA = 1   #   0 < GAMMA 
J = 1     # 
KBT = J*0.001*0
dt = 0.001  #picoseconds?

consts = Consts(ALPHA=0.1,GAMMA=1,J = 1,KBT=0.1*J*0,B = np.array([0,0,1],dtype=np.float64),d_z = 0,magMom = 5.788*10**-2)

# -----------------

def spinlattice(N, M, random = False,start = np.array([0,0,1])):
    """
    Generate/initialize NxM spinlattice 

    returns:
        spinlattice : np.ndarray(shape = (N, N, 3))
    """
    initial = np.zeros(shape = (N,M,3))
    if random:
        randoms = np.random.randn(N*M,3)
        randoms = randoms/np.linalg.norm(randoms, axis = -1)[...,np.newaxis]
        randoms = np.reshape(randoms,(N,M,3))
        initial[:,:] = randoms
        print(initial.shape)
    else:                  
        initial[:,:] = start
    return initial




@fast.njit(cache = True)
def dtSpin(S : np.ndarray, F : np.ndarray,GAMMA,ALPHA):
    prefac = -GAMMA/(1+ALPHA**2) 
    terms = np.cross(S,F)+ALPHA*np.cross(S,np.cross(S,F)) 
    return prefac*terms

@fast.njit(parallel = True)
def effectiveFlatticeNonPeriodic(lattice,magMom,d_z, B: np.ndarray, J : float, dt : float):
    yMax = len(lattice[0])
    xMax = len(lattice[:,0])
    ez = np.array([0,0,1])
    nextLattice = lattice.copy()
    for y in fast.prange(yMax):
        for x in range(xMax):
            
            anis = 2*d_z*np.linalg.norm(lattice[y,x],ord = 1)*ez
            
            spincoupling = J * (lattice[y,x+1] + lattice[y,x-1] + lattice[y+1,x] + lattice[y-1,x]) 
            
            Fj = B + spincoupling/magMom + anis/magMom
            
            spinForce = dtSpin(lattice[y,x],Fj)

            for i in range(3):
                nextLattice[y,x,i] = lattice[y,x,i] + dt*spinForce[i]

            nextLattice[y,x] = nextLattice[y,x]/np.linalg.norm(nextLattice[y,x])
    
    return nextLattice

@fast.njit(cache = True)
def latticeChange(lattice, randnums, dt, xmax, ymax, C : Consts,notPeriodic):
    ez = np.array([0,0,1])
    tempLattice = np.zeros(shape = (ymax+notPeriodic,xmax+notPeriodic,3))
    res = np.zeros(shape = (ymax,xmax,3))
    if notPeriodic:
        tempLattice[:-1,:-1] = lattice
    else:
        tempLattice[:,:] = lattice

    randomscaling = np.sqrt(2*C.ALPHA*C.KBT/(C.GAMMA*C.magMom*dt))
    for y in range(ymax):
        
        for x in range(xmax):

            anis = 2*C.d_z*np.linalg.norm(tempLattice[y,x],ord = 1)*ez
            
            spincoupling = J * (tempLattice[y,x+1] + tempLattice[y,x-1] + tempLattice[y+1,x] + tempLattice[y-1,x]) 
            
            Fj = C.B + spincoupling/C.magMom + anis/C.magMom

            spinChange = dtSpin(lattice[y,x],Fj,C.GAMMA,C.ALPHA)
            for i in range(3):
                res[y,x,i] = spinChange[i] + randnums[y,x,i]*randomscaling
    return res




#@fast.njit(cache = True)
def HeunsMethod3dLattice(lattice : np.ndarray, dt : float, steps :int, C : Consts,notPeriodic = True):
    xMax = len(lattice[0]) 
    yMax = len(lattice[:,0]) 
    lattices = np.zeros(shape =(steps+1,yMax,xMax,3), dtype = np.float64)
    lattices[0] = lattice

    mean = np.array([0,0,0])
    covs = np.eye(3)
    for t in range(steps):
        randnums = np.random.multivariate_normal(mean,covs,size = (yMax,xMax))
        
        F = latticeChange(lattices[t],randnums,dt,xMax,yMax,C,notPeriodic)

        yp  = normalize(lattices[t] + dt*F)

        predictorF = latticeChange(yp,randnums,dt,xMax,yMax,C,notPeriodic)

        lattices[t+1] = normalize(lattices[t] + dt/2*(F + predictorF))

    return lattices

def normalize(vecs):
    return vecs/np.linalg.norm(vecs,axis=-1)[...,np.newaxis]
    

def atoms1d(atoms,steps,C:Consts,random = False,c = False):
    log.info("Running...")
    C.ALPHA=0
    s = spinlattice(atoms,1,start=np.array([0,0,1])/np.linalg.norm([0,0,1]))
    if c:
        s[0,0] = np.array([0.2,0,1])/np.linalg.norm([0.2,0,1])
    A = HeunsMethod3dLattice(s,0.001,steps,C)
    log.info(f"Heuns done! Shape = {A.shape}")
    
    if c:
        ts = np.arange(steps+1)*dt
        fig,axes = plt.subplots(3)
        for i in range(0,atoms,3):
            axes[0].plot(ts,A[:,i,0,0])
            axes[1].plot(ts,A[:,i,0,1])
            axes[2].plot(ts,A[:,i,0,2])
    else:
        fig,axes = plt.subplots(1,3)
        for i,ax in enumerate(axes):
            ax.imshow(A[::25,:,0,i])

    plt.show()
#atoms1d(100,100000,consts,random=False)

def a(steps, consts : Consts,b = False):
    s = spinlattice(1,1,0)
    dt = 0.001
    if not b:
        consts.ALPHA=0
    A = HeunsMethod3dLattice(s,dt,steps,consts,True)
    fig,axes = plt.subplots(1,3,sharey=True)
    ts = np.arange(steps+1)*dt
    for i,ax in enumerate(axes):
        ax.plot(ts,A[:,0,0,i])
    if b:
        def expCos(x,w,T):
            return np.exp(-x*T)*np.cos(w*x)
        from scipy.optimize import curve_fit
        (w,T),_ = curve_fit(expCos,ts,A[:,0,0,0],p0 = (3,1))
        axes[0].plot(ts,expCos(ts,w,T), label = "curvefitted",linewidth = 0.8)
        print(w)
        axes[0].plot(ts,expCos(ts,w,(consts.ALPHA*w)), label = "theroetival",linewidth = 1)
        axes[0].legend()

    plt.show()

consts.ALPHA = 1

atoms1d(atoms=30,steps=1500, C = consts, c = True)



def plotQuivers(A,M,N):
    plottingPlane = (0,1,2)   # (0, 1, 2) corresponds to the coordinates (x, y, z) respectively
    fig, ax = plt.subplots(1)
    X = np.arange(M)
    Y = np.arange(N)
    X,Y = np.meshgrid(X,Y)

    Q = ax.quiver(X,Y,A[0,:,:,plottingPlane[0]],A[0,:,:,plottingPlane[1]], A[0,:,:,plottingPlane[2]])

    def update_quiver(i,Q,A):
        """
        updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """
        U = A[i,:,:,plottingPlane[0]]
        V = A[i,:,:,plottingPlane[1]]
        C =  A[i,:,:,plottingPlane[2]]

        Q.set_UVC(U,V,C)
        return Q,
    log.    info("Creating animation")
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
