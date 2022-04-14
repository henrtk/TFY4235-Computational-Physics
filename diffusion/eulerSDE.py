from re import X
from time import time
from numba import njit,vectorize, float64,prange
import matplotlib.pyplot as plt
import numpy as np

class Particle:
    def __init__(self,t,x0,period,dT):
        self.t, self.x, self.period, self.dT = t, x0, period, dT
        self.traj = np.array([x0])
    def update(self,steps):
        self.traj = np.append(self.traj,forwardEulerTraj(self.x,steps,self.period)[1:])
        self.t += self.dT*steps
        self.x = self.traj[-1]

# Globals, reduced units

steps = 40000000
_alpha = 0.3
_oalpha = 1-_alpha
_D = 0.005
_delT = 0.5*10**-3


def setDFromSi(eta,r,dU,kbT):
    """
    Sets D based on given physical parameters in SI-units
    """
    D = kbT/(6*np.pi*eta*r*dU)
    return D
def translateReducedUnits(x,t,delT,dU,gamma,L):
    xTrue = x*L
    omega = dU/(gamma*L*L)
    tTrue = t/omega
    delTtrue = delT/omega
    UTrue = dU*U(x,t)
    return xTrue,tTrue,delTtrue,UTrue

@vectorize([float64(float64,float64,float64)])
def U(x_n,t_n,period = 3):
    tcheck = np.abs(t_n%period)<3/4*period      # check if potential should be on
    periodicPos = np.abs(x_n%1)                 # find position within potential
    if (periodicPos<_alpha):                    # handle time-dependence
        return periodicPos*tcheck/_alpha
    else:
        return (1-periodicPos)*tcheck/(_oalpha)

@vectorize([float64(float64,float64,float64)])
def F(x_n : np.float64, t_n : np.float64, period:np.float64) -> np.float64:
    tcheck = (np.abs(t_n%period)<(3/4*period))  # check if potential should be on
    periodicPos = np.abs(x_n%1)                 # find position within potential
    if (periodicPos<_alpha):                    # handle time-dependence
        return -tcheck/_alpha
    else:
        return tcheck/(_oalpha)

def _boxMuller(pairedUniformArray):
    R = np.sqrt(-2*np.log(pairedUniformArray[::2]))
    ct = np.cos(2*np.pi*pairedUniformArray[1::2])
    st = np.sin(2*np.pi*pairedUniformArray[1::2])
    result =  np.append(R*ct,R*st)
    return result

def demonstrateBM(samples):
    plt.title(f"Demonstration of Box-Muller algorithm, n = {samples} ")
    plt.hist(_boxMuller(np.random.uniform(size=samples)),bins = 100,density = True, label = "BM-results")
    plt.plot(np.linspace(-4,4),1/np.sqrt(2*np.pi)*np.exp(-(np.linspace(-4,4)**2/2)), label = "Standard normal distribution")
    plt.legend()
    plt.show()


def plotPotForceX(x_max):
    x = np.linspace(-1,x_max,10000)
    # want to plot the potentials in the state ON, hence 0.99
    plt.plot(x,U(x,-0.001))
    plt.plot(x,F(x,-0.001))
    plt.show()

@njit
def testTimeStep(t): 
    S = np.sqrt(2*_D)
    invSsq = 1/(2*_D)
    return _alpha*S*(np.sqrt(4+invSsq)-2)>5*np.sqrt(t) # factor 5 to ensure >> condition

@njit
def forwardEulerTraj(x_0 : np.float64, steps : int, period : np.float64) -> "np.ndarray(shape = steps, dtype=np.float64)" :
    """
    Forward Euler for diffusion SDE
    
    returns: trajectory -> np.ndarray(shape = steps, dtype = np.float64)
     """
    x = np.empty(steps+1)
    rands = np.random.standard_normal(steps)
    x[0] = x_0
    ts = np.linspace(0,_delT*steps,steps+1)
    for i in range(steps):
        x[i+1] = x[i] + F(x[i],ts[i],period)*_delT + np.sqrt(2*_D*_delT)*rands[i]
    return x

@njit
def forwardEulerEndp(x0 : np.float64, steps : int, period : np.float64):
    """
    Forward Euler with no trajectory generation.

    returns: endpoint -> np.float64
    """
    x = x0
    ts = np.linspace(0,_delT*steps,steps+1) # fairly sure this prevents accumulation of fp-error in t
    for i in range(steps):
        x += F(x,ts[i],period)*_delT + np.sqrt(2*_D*_delT)*np.random.standard_normal()
    return x

@njit
def simulateParticles(n,iterations,period):
    """
    Simulates n particles using #iterations = iterations
    in the Forward Euler scheme with given period.

    Return: mean and non-empirical std.
    (note: std is biased because numba doe not like the correct
     ddof = 1 argument. Still gives a decent estimate.)
    """
    partpos = []
    for i in range(n):
        partpos.append(forwardEulerEndp(0,iterations,period))
    partpos = np.asarray(partpos)
    return np.array([np.mean(partpos), partpos.std()])

@njit
def simulateParticlesDetailed(n,iterations,period):
    """
    simulates n particles 
    """
    partpos = []
    for i in range(n):
        partpos.append(forwardEulerEndp(0,iterations,period))
    partpos = np.asarray(partpos)
    return np.array([np.mean(partpos), partpos.std()]), partpos


@njit(parallel = True)
def datagen(start,end,fineness,particles = 10,iterations = 10000):
    """
    Generates statistics for uniformally spaced periods
    using parameter particles as #simulations for every
    unique period. Parallelized using numba.
    """
    # generate period landscape
    area  = np.linspace(start,end,fineness)
    # generate some random seeds for reproduction and secure randomization in parallel.
    seeds = np.random.randint(low = 0, high = 10000, size = fineness)
    datas = []
    # initiate parallel simulation of particle drift
    for i in prange(fineness): 
        np.random.seed(seeds[i])
        mu,std = simulateParticles(particles,iterations,area[i])
        datas.append((area[i],mu,std))
        # feedback to check progress during runtime, assumes NUMBA_NUM_THREADS is set to 12
        if i % 12 == 0: 
            print("Progress: ", i-12 , "simulations done")
    return datas, seeds

def plotParticleTrajectory(xs,delT,period, color = None):
    ts = np.linspace(0,(len(xs)-1)*delT,len(xs))
    plt.plot(ts,xs, label = f"{round(period,2)}",color = color)
    plt.text(ts[-1]+ts[-1]*0.01,xs[-1],s = str(round(period,1)))
    return

def downhillSimp(errormax,partIterations,particles = 30, maxiter = 500):
    """
    Implements something similar to downhill simplex D = 2 in 1d.
    Maximizes particle position wrt. flashing period.

    """
    Ts = np.array([15,20,100.0])
    fs = np.array([simulateParticles(particles,partIterations,Ts[0]),simulateParticles(particles,partIterations,Ts[1]),simulateParticles(particles,partIterations,Ts[2])])
    start = time()
    i = 0
    print(Ts)
    while abs(max(fs[:,0])-min(fs[:,0])) > (errormax) or max(fs[:,0])<5:
        fs_sort = np.argsort(fs[:,0]) #lowest first
        if not fs_sort[1] == 1:
            mini = fs_sort[0]
            Tmean = (Ts[fs_sort[2]]+Ts[fs_sort[1]])/2 
            if (Tmean + 0.9*(Tmean-Ts[mini]))>0:
                Ts[mini] = Tmean + 0.9*(Tmean-Ts[mini])
            else: 
                Ts[mini] = 2*_delT
            fs[mini,:] = simulateParticles(particles,partIterations,Ts[mini])

            plt.errorbar(Ts[mini],fs[mini][0],fs[mini][1],fmt="o")
            plt.text(Ts[mini],fs[mini][0],i)
        else:
            mini = fs_sort[0]
            Tmean = (Ts[fs_sort[2]]+Ts[fs_sort[1]])/2 
            if (Tmean + 0.5*(Tmean-Ts[mini]))>0:
                Ts[mini] = Tmean + 0.5*(Tmean-Ts[mini])
            else: 
                Ts[mini] = 2*_delT
            fs[mini,:] = simulateParticles(particles,partIterations,Ts[mini])
            plt.errorbar(Ts[mini],fs[mini][0],fs[mini][1],fmt="o")
            plt.text(Ts[mini],fs[mini][0],i)
        i+=1
        print(i, fs[:,0], Ts) 
        if maxiter <= i:
            break
    for T,f in zip(Ts,fs):
        plt.errorbar(T,f[0],f[1],fmt="o")
    plt.show()
    print("End at time = ", time()-start,"Period found",Ts)
    return Ts

if __name__ == "__main__":
    print("delT small?", testTimeStep(_delT))
    downhillSimp(4,steps,20)
