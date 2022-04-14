import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import eulerSDE as d
from scipy.optimize import curve_fit


def normal(x,mu,sigma):
    return np.exp(-((x-mu)/sigma)**2/2)/(sigma*np.sqrt(2*np.pi))


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
 
    x = np.linspace(min(particlepos)-10,max(particlepos)+10,10000)
    axs[1].pcolormesh(x,[0,1],[100*normal(x,mu,sigma),100*normal(x,mu,sigma)])#density = True, bins = (diff,1))
    
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
