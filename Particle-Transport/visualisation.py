import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from eulerSDE import U, get_D, simulateParticlesDetailed, _boxMuller, F
from scipy.optimize import curve_fit
import logging as log

def normal(x,mu,sigma):
    return np.exp(-((x-mu)/sigma)**2/2)/(sigma*np.sqrt(2*np.pi))

def boltz(x):
    """
    Returns the boltzman distribution of the position as a function of its potential.
    Not normalized! Time parameters of U set to yield the potential turned ON.
    """
    boltzfactor = -U(x,0.01,2*0.01)/get_D()
    return np.exp(boltzfactor)/(1-np.exp(-1/get_D()))

def histogramtransport(particleDist,fine = 5):
    """
    Create and plot fancy colourbased histogram  from particle positions
    with comparative fitted normal distirbution.
    """
    fig, axs = plt.subplots(ncols = 1,nrows = 2, sharex=True)

    # generate data and histogram bin midpoints as x-positions to fit normal distribution
    hist = np.histogram(particleDist,bins = "auto", density = True)
    mids = [(i+j)/2 for i,j in zip(hist[1][1:],hist[1][:-1])]
    # fit normal distribution
    (mu,sigma), _ = curve_fit(normal,mids,hist[0],p0 = (mids[len(mids)//2],10))
    
    # create and plot boltz-distribution
    
    # decide approximate concentration by scaling particle concentrations by the fitted normal distribution  
    

    # plot numerically simulated particle distributions in a style similar to original paper.
    minTravel, maxTravel = int(min(particleDist)),int(max(particleDist))
    # create x-axis coords for plotting
    x = np.linspace(minTravel-1,maxTravel+1,11*(maxTravel-minTravel+2)+1)
    boltzDist = boltz(x)

    envelopedBoltzDist = [normal(x,mu,sigma)*boltzDist,normal(x,mu,sigma)*boltzDist]  
    axs[1].pcolormesh(x,[0,1],envelopedBoltzDist)

    diff = np.linspace(minTravel-0.5,maxTravel+0.5,fine*(maxTravel-minTravel+2))
    dummyY = np.zeros(len(particleDist))
    h = axs[0].hist2d(particleDist,dummyY,density = True,bins = (diff,1))
    
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[1].set_xlabel("x/L")
    axs[1].set_ylabel("Normal distribution")
    axs[0].set_ylabel("Numerical results")
    axs[0].set_title(f"Numerical results vs normal distribution, $N = {len(particleDist)}$ particles")
    fig.colorbar(h[3],ax = axs.ravel().tolist(), aspect = 10)
    plt.show()

def histogramtransport2(particleDist):
    """
    Create and plot histogram from particle positions
    with fitted normal distribution overlayed.
    """
    hist = np.histogram(particleDist,bins = "auto", density = True)
    mids = [(i+j)/2 for i,j in zip(hist[1][1:],hist[1][:-1])]
    diff = np.linspace(int(min(particleDist))-0.5,int(max(particleDist))+0.5,(-int(min(particleDist))+int(max(particleDist)+2)))
    (mu,sigma), _ = curve_fit(normal,mids,hist[0],p0 = (mids[len(mids)//2],1))

    fig,ax = plt.subplots()
    ax.hist(particleDist,density = True,bins = diff)
    x = np.linspace(min(particleDist)-5,max(particleDist)+5,10000)
    ax.plot(x,normal(x,mu,sigma), color = "green")
    ax.fill(x,normal(x,mu,sigma),alpha = 0.3, color = "green")#density = True, bins = (diff,1))

    ax.set_yticks([])
    ax.set_xlabel("x/L")
    ax.set_title("Numerical results vs normal distribution")
    plt.show()

def boltzmannDistViz(n,dt =10**-4,iterations = 10000):
    """
    Create and plot fancy colourbased histogram from particle positions
    with comparative fitted normal distirbution.

    params:
        n : int -> # of particles to simulate
        dt: float -> Their timestep
        iterations: the amount of iterations to do for each particle

    returns:
        void (interactive plot)
    """
    _, particleDist = simulateParticlesDetailed(n,iterations,period = dt*2,dt = dt)
    hist = np.histogram((particleDist)%1,bins = "auto", density = True)

    interval = np.linspace(0,1,n)
    fig,axs = plt.subplots()
    axs.hist(particleDist%1,density = True,bins =20)

    # Normalize and plot boltzmann distribution using a simple riemann sum.
    axs.plot(interval,boltz(interval)/(np.sum(boltz(interval))*(interval[1]-interval[0])))

    axs.set_xlabel("x/L")
    axs.set_title("Numerical results vs Boltzmann distribution")
    plt.show()

def plotParticleTrajectory(trajectory,dt,period, color = None):
    ts = np.linspace(0,(len(trajectory)-1)*dt,len(trajectory))
    plt.plot(ts,trajectory, label = f"{round(period,2)}",color = color)
    #plt.text(ts[-1]+ts[-1]*0.01,trajectory[-1],s = str(round(period,1)))
    return

def demonstrateBM(samples):
    plt.title(f"Demonstration of Box-Muller algorithm, n = {samples} ")
    plt.hist(_boxMuller(np.random.uniform(size=samples)),bins = 100,density = True, label = "BM-results")
    plt.plot(np.linspace(-4,4),1/np.sqrt(2*np.pi)*np.exp(-(np.linspace(-4,4)**2/2)), label = "Std. normal distribution")
    plt.legend()
    plt.show()

def plotPotForceX(x_max):
    x = np.linspace(-1,x_max,10000)
    # want to plot the potentials in the state ON, hence 0.99
    plt.plot(x,U(x,-0.001))
    plt.plot(x,F(x,-0.001))
    plt.show()

if __name__ == "__main__":

    boltzmannDistViz(n = 100_000, dt = 10**-4, iterations=100_000)

    #_,pos = simulateParticlesDetailed(20000,631400,5,10**-3)
    #histogramtransport(pos, fine = 10)
