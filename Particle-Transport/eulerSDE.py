from time import time
from numba import njit,vectorize, float64,prange, pycc
import matplotlib.pyplot as plt
import numpy as np

# Globals (regrettably), reduced units
_alpha = 0.2
_oalpha = 1-_alpha
_D = 0.1
# ----------------------

class Particle:
    def __init__(self,t,x0,period,dT) -> None:
        self.t, self.x, self.period, self.dT = t, x0, period, dT
        self.traj = np.array([x0])
    def update(self,steps):
        self.traj = np.append(self.traj,forwardEulerTraj(self.x,steps,self.period)[1:])
        self.t += self.dT*steps
        self.x = self.traj[-1]

class ParticleEnsemble:
    # Discontinued
    def __init__(self,t,x0,period,dT,numParticles : int = 5) -> None:
        self.particles = [Particle(t,x0,period,dT) for i in range(numParticles)]
    def parallelUpdate():
        pass

def setParamsSI(r,viscosity,dU,L,tHat,xHat):
    gamma = 6*np.pi*viscosity*r
    t = tHat*gamma*L**2/dU
    x = L*xHat
    kbT = _D*dU
    return t, x, kbT

def get_D():
    return _D

def translateReducedUnits(x,t,delT,dU,gamma,L):
    xTrue = x*L
    omega = dU/(gamma*L*L)
    tTrue = t/omega
    delTtrue = delT/omega
    UTrue = dU*U(x,t)
    return xTrue,tTrue,delTtrue,UTrue

@vectorize([float64(float64,float64,float64)], nopython = True,cache = True)
def U(x_n,t_n,period = 3):
    tcheck = np.abs(t_n%period) < (1/4*period)  # check if potential should be on
    
    periodicPos = (x_n)%1                 # find position within potential
    
    if (periodicPos<_alpha):                    # handle time-dependence
        return periodicPos*tcheck/_alpha        # boolean multiplication!
    else:
        return (1-periodicPos)*tcheck/(_oalpha)

@vectorize([float64(float64,float64,float64)], nopython = True, cache = True)
def F(x_n : np.float64, t_n : np.float64, period:np.float64) -> np.float64:
    tcheck = (np.abs(t_n%period)<(1/4*period))  # check if potential should be on
    periodicPos = np.abs((x_n)%1)                 # find position within potential
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

@njit(cache = True)
def testTimeStep(t): 
    S = np.sqrt(2*_D)
    invSsq = 1/(2*_D)
    return _alpha*S*(np.sqrt(4+invSsq)-2)/(5*np.sqrt(t))# factor 5 to ensure >> condition

@njit(cache = True)
def forwardEulerTraj(x_0 : np.float64, steps : int, period : np.float64, dt : np.float64 = 10**-3) -> "np.ndarray(dtype=np.float64)" :
    """
    Forward Euler for diffusion SDE
    
    returns: trajectory -> np.ndarray(shape = steps, dtype = np.float64)
     """
    x = np.empty(steps+1)

    rands = np.random.standard_normal(steps)
    x[0] = x_0

    ts = np.linspace(0,dt*steps,steps+1)

    for i in range(steps):
        x[i+1] = x[i] + F(x[i],ts[i],period)*dt + np.sqrt(2*_D*dt)*rands[i]
    return x

@njit(cache = True)
def forwardEulerEndp(x0 : np.float64, steps : int, period : np.float64, dt : np.float64 = 10**-3):
    """
    Forward Euler with no trajectory generation.

    returns: 
        endpoint -> np.float64
    """
    x = x0
    if steps < 100_000_000: #Memory consideration! about 8GB generated when run at this time
        ts = np.linspace(0,dt*steps,steps+1) # fairly sure this prevents accumulation of fp-error in t

        for i in range(steps):
            x += F(x,ts[i],period)*dt + np.sqrt(2*_D*dt)*np.random.standard_normal()

    else:
        t = 0
        for i in range(steps):
            x += F(x, t, period)*dt + np.sqrt(2*_D*dt)*np.random.standard_normal()
            t += dt
    return x

@njit(parallel = True, cache = True)
def simulateParticles(n,iterations,period,dt):
    """
    Simulates n particles using #iterations = iterations
    in the Forward Euler scheme with given period.
    Parallelized using numba.

    returns: 
        np.ndarray([mean,  empirical st. deviation])
    """
    partpos = np.empty(n)
    for i in prange(n):
        partpos[i] = forwardEulerEndp(0,iterations,period,dt)
    return np.array([partpos.mean(), partpos.std()])

@njit(cache = True, parallel =True)
def simulateParticlesDetailed(n,iterations,period,dt):
    """
    Simulates n particles using #iterations = iterations
    in the Forward Euler scheme with given period.
    Parallelized using numba.

    Return: 
        np.ndarray([mean, std]),
        list([end positions])
    """
    if not testTimeStep(dt):
        print("Uh oh, chance of particles jumping across potential barriers")
    partpos = np.empty(n)
    for i in prange(n):
        partpos[i] = forwardEulerEndp(-0.0,iterations,period,dt)
    return np.array([partpos.mean(), partpos.std()]), partpos

@njit(parallel = True, cache = True)
def datagen(start,end,fineness,particles = 10,iterations = 10000, dt = 10**-4):
    """
    Generates statistics for uniformally spaced periods
    using parameter particles as #simulations for every
    unique period.

    returns:
        (list([period, mean, std,speedmean,speedstd()]), np.ndarray([seeds]))
    """
    # generate period landscape
    area  = np.linspace(start,end,fineness)
    datas = np.empty(shape = (fineness,5))
    # initiate parallel simulation of particle drift
    for i in prange(fineness): 
        (mu,std),posits = simulateParticlesDetailed(particles,iterations,area[i],dt)
        speeds = np.empty(particles)
        for j, pos in enumerate(posits):
            speeds[j] = pos/(iterations*dt)
        datas[i] = np.array([area[i],mu,std,speeds.mean(),speeds.std()])
        if i % 12 == 0: 
            print("Progress: ", i , "simulations done")
    return datas

def downhillSimp(errormax,partIterations,dt,particles = 30, maxiter = 500):
    """
    Implements something similar to downhill simplex D = 1 in 1d.
    Maximizes particle position wrt. flashing period.
    Not used in the report, but used to quickly find the area to look for the best flashing period.

    """
    Ts = np.array([15,20,100.0])
    fs = np.array([simulateParticles(particles,partIterations,Ts[0],dt),simulateParticles(particles,partIterations,Ts[1],dt),simulateParticles(particles,partIterations,Ts[2],dt)])
    start = time()
    i = 0
    print(Ts)
    while abs(max(fs[:,0])-min(fs[:,0])) > (errormax) or max(fs[:,0])<5:
        fs_sort = np.argsort(fs[:,0]) #lowest first
        
        mini = fs_sort[0]
        Tmean = (Ts[fs_sort[2]]+Ts[fs_sort[1]])/2 

        if not fs_sort[1] == 2: # The middle period is NOT in the middle. This indicate

            if (Tmean + 0.9*(Tmean-Ts[mini]))>0: #prevents negative numbers
                Ts[mini] = Tmean + 0.9*(Tmean-Ts[mini])
            else: 
                Ts[mini] = 10*dt
            fs[mini,:] = simulateParticles(particles,partIterations,Ts[mini],dt)

            plt.errorbar(Ts[mini],fs[mini][0],fs[mini][1],fmt="o")
            plt.text(Ts[mini],fs[mini][0],i)
        else:
            
            if (Tmean + 0.5*(Tmean-Ts[mini]))>0: # again, prevent negative period being generated
                Ts[mini] = Tmean + 0.5*(Tmean-Ts[mini])
            else: 
                Ts[mini] = 10*dt # set to a small number instead

            fs[mini,:] = simulateParticles(particles,partIterations,Ts[mini],dt)
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

def plotFromPickle(filename1, filename2):
    """ 
    Plot saved generated data in file "filename1/2".
    """
    import pickle
    with open(filename1,"rb") as f:
        data = pickle.load(f)
    with open(filename2,"rb") as f:
        data2 = pickle.load(f)
    
    plt.plot(data[:,0],data[:,3], label = "Drift velocity particle 1", color = "r")
    plt.plot(3*data2[:10,0],data2[:10,3],label = "Drift velocity particle 2", color = "b")

    plt.xlabel("Flashing period")
    plt.ylabel("Reduced unit drift velocity")
    plt.fill_between(data[:,0], data[:,3]-data[:,4],data[:,3]+data[:,4], alpha = 0.3, color ="r")
    plt.fill_between(3*data[:10,0], data2[:10,3]-data2[:10,4],data2[:10,3]+data2[:10,4], alpha = 0.3, color ="b")
    plt.vlines(0.55,0.01,0.035)
    plt.vlines(1.65,0.01,0.035)
    plt.legend()
    plt.show()
if __name__ == "__main__":
    steps = 200_000_000 #uh oh be careful with the memory
    _delT = 0.5*10**-3

    plotFromPickle(filename2="Data for particle 1, t-hat is 400 000, dt 10-4",filename1= "Data for particle 1 (actual 1), t-hat is 400 000, dt 10-4")
    #print("delT small?", testTimeStep(_delT))
    #downhillSimp(3,steps,_delT,50)
    #data = np.asarray(datagen(0.1,3,30,24,steps,dt=_delT))
    #for i in data:
        #print(i)
    import pickle

    #with open("Data for particle 1 (actual 1), t-hat is 400 000, dt 10-4", "wb") as f:
     #   pickle.dump(data,f)
    #plt.plot(data[:,0],data[:,3], label = "Average particle reduced unit drift velocity", color = "r")
    #plt.xlabel("Flashing period")
    #plt.ylabel("Reduced unit drift velocity")
    #plt.fill_betweenx(data[:,0],data[:,3]+data[:,4], data[:,3]-data[:,4], alpha = 0.3, color ="r")
    #plt.show()