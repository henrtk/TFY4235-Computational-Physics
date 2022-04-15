import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import eulerSDE as d
from scipy.optimize import curve_fit


def normal(x,mu,sigma):
    return np.exp(-((x-mu)/sigma)**2/2)/(sigma*np.sqrt(2*np.pi))
def boltz(x):
    boltzfactor = -d.U(x,0.01,2*0.01)/d.get_D()
    return np.exp(boltzfactor)/(1-np.exp(-1/d.get_D()))

def histogramtransport(particlepos,fine = 5):
    """
    Create and plot fancy colourbased histogram  from particle positions
    with comparative fitted normal distirbution.
    """
    hist = np.histogram(particlepos,bins = "auto", density = True)
    mids = [(i+j)/2 for i,j in zip(hist[1][1:],hist[1][:-1])]
    parts = len(particlepos)
    diff = np.linspace(int(min(particlepos))-0.5,int(max(particlepos))+0.5,fine*(-int(min(particlepos))+int(max(particlepos)+2)))
    (mu,sigma), _ = curve_fit(normal,mids,hist[0],p0 = (mids[len(mids)//2],1))
    
    fig,axs = plt.subplots(ncols = 1,nrows = 2, sharex=True)
    h = axs[0].hist2d(particlepos,np.zeros(parts),density = True,bins = (diff,1))
    x = np.linspace(int(min(particlepos))-1,1+int(max(particlepos)),11*(-int(min(particlepos))+int(max(particlepos))+2)+1)
    mock= boltz(x)
    
    axs[1].pcolormesh(x,[0,1],[normal(x,mu,sigma)*mock,mock*normal(x,mu,sigma)])#density = True, bins = (diff,1))
    
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[1].set_xlabel("x/L")
    axs[0].set_title("Numerical results vs normal distribution")
    fig.colorbar(h[3],ax = axs.ravel().tolist(), aspect = 10)
    plt.show()

def histogramtransport2(particlepos):
    """
    Create and plot histogram from particle positions
    with fitted normal distribution overlayed.
    """
    hist = np.histogram(particlepos,bins = "auto", density = True)
    mids = [(i+j)/2 for i,j in zip(hist[1][1:],hist[1][:-1])]
    diff = np.linspace(int(min(particlepos))-0.5,int(max(particlepos))+0.5,(-int(min(particlepos))+int(max(particlepos)+2)))
    (mu,sigma), _ = curve_fit(normal,mids,hist[0],p0 = (mids[len(mids)//2],1))

    fig,ax = plt.subplots()
    ax.hist(particlepos,density = True,bins = diff)
    x = np.linspace(min(particlepos)-5,max(particlepos)+5,10000)
    ax.plot(x,normal(x,mu,sigma), color = "green")
    ax.fill(x,normal(x,mu,sigma),alpha = 0.3, color = "green")#density = True, bins = (diff,1))

    ax.set_yticks([])
    ax.set_xlabel("x/L")
    ax.set_title("Numerical results vs normal distribution")
    plt.show()

def boltzmannDistViz(n,dt =10**-4,iterations = 100000):
    """
    Create and plot fancy colourbased histogram  from particle positions
    with comparative fitted normal distirbution.
    """
    _, particlepos = d.simulateParticlesDetailed(n,iterations,period = dt*2,dt = dt)
    hist = np.histogram(particlepos,bins = "auto", density = True)

    diff = np.linspace(-0.4,0.05,n*10)
    fig,axs = plt.subplots()
    axs.hist(particlepos,density = True,bins = hist[1])
    axs.plot(diff,boltz(diff)/(np.cumsum(boltz(diff)*(diff[1]-diff[0]))[-1]))

    axs.set_xlabel("x/L")
    axs.set_title("Numerical results vs normal distribution")
    plt.show()

def plotParticleTrajectory(trajectory,dt,period, color = None):
    ts = np.linspace(0,(len(trajectory)-1)*dt,len(trajectory))
    plt.plot(ts,trajectory, label = f"{round(period,2)}",color = color)
    plt.text(ts[-1]+ts[-1]*0.01,trajectory[-1],s = str(round(period,1)))
    return

def demonstrateBM(samples):
    plt.title(f"Demonstration of Box-Muller algorithm, n = {samples} ")
    plt.hist(d._boxMuller(np.random.uniform(size=samples)),bins = 100,density = True, label = "BM-results")
    plt.plot(np.linspace(-4,4),1/np.sqrt(2*np.pi)*np.exp(-(np.linspace(-4,4)**2/2)), label = "Standard normal distribution")
    plt.legend()
    plt.show()

def plotPotForceX(x_max):
    x = np.linspace(-1,x_max,10000)
    # want to plot the potentials in the state ON, hence 0.99
    plt.plot(x,d.U(x,-0.001))
    plt.plot(x,d.F(x,-0.001))
    plt.show()

if __name__ == "__main__":
    _,pos = d.simulateParticlesDetailed(20000,631400,5,10**-3)
    histogramtransport(pos, fine = 10)
    pass