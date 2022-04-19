from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib.cm as cm
from numba import njit,vectorize,prange
import seaborn as sns



def gammaDecay(decaySpectrum : np.ndarray) -> np.float64:
    draw = np.random.random_sample()
    for E, P in decaySpectrum:
        if draw <= P:
            return E
    else:
        return decaySpectrum[1,-1]
@njit
def chooseTrajectory(cdf : np.ndarray) -> np.float64:
    """
    Chooses a random trajectory based on numbas Mersenne Twister.
    """
    draw = np.random.random_sample()
    theta = np.linspace(0,np.pi*2,len(cdf))
    theta+= (theta[1]-theta[0])/2
    for i,highEnd in enumerate(cdf): #Optimalization: Use binary search! Scratch that, basicly same time used, bottleneck elsewhere...
        if draw <= highEnd:
            return theta[i]
    else:
        return cdf[0]



@vectorize(nopython = True, cache = True)
def _kleinNishina(Eg : np.float64, theta :np.float64) -> np.float64 :
    """
    Calculates variable dependent part of the scattering cross-section in a direction theta
    To be used with normalization, hence the disregard for prefactors
    """
    electronRestMassEnergy = 511 #keV   
    k = Eg/electronRestMassEnergy
    sqTerm = 1/(1+k*(1-np.cos(theta))) # beware the floating point error!
    finalPar = sqTerm+1+k*(1-np.cos(theta))-np.sin(theta)**2
    sqTerm*=sqTerm
    return sqTerm*finalPar

@njit
def PDF(Eg,fineness = 10000):
    theta = np.linspace(0,2*np.pi,fineness)
    dtheta = theta[1]-theta[0]
    kleinNishina = _kleinNishina(Eg,theta)
    pdf = kleinNishina/(np.sum(kleinNishina)*dtheta)
    return pdf
@njit
def CDF(Eg,fineness = 10000):
    return np.cumsum(PDF(Eg,fineness))*np.pi*2/fineness

@njit(parallel = True,cache = True)
def monteCarloDirection(Eg,fineness):
    cdf = CDF(Eg,fineness)
    thetas = np.empty(10000000)
    for i in prange(10000000):
        thetas[i] = chooseTrajectory(cdf)
    return thetas

@njit
def randomPhotoElectric(Eg,Z,dx):
    """
    returns True if MonteCarlo-photoelectric interaction has taken place 
    """
    sigma = 3*10**12*Z**4*(10**-3/Eg)**3.5 #OBS, Eg is in keV! dx-scale must make up for this
    p = 1-np.exp(-sigma*dx)
    return p >= np.random.random_sample()

@njit
def postComptonEnergy(Eg,theta):
    electronRestMassEnergy = 511 #keV   
    k = Eg/electronRestMassEnergy
    return Eg/(1+k*(1-np.cos(theta)))

@njit(cache = True)
def monteCarloPhotoscattering(E0,theta0,Z,dx,steps,anglularfineness = 1000):
    Es = np.empty(steps+1)
    posits = np.empty((steps+1,2))              #xs and ys
    theta = theta0
    y0 = 0#(np.random.random_sample()+np.random.choice(np.array([0,7]))-0.5)*5*dx 
    Es[0], posits[0] = E0, np.array([0,y0])
    for i in range(steps):
        l = dx*np.random.random_sample()
        if randomPhotoElectric(Es[i],Z,dx):
            Es, posits = Es[:i], posits[:i]
            break
        elif posits[i][0] < 0:
            cont = 100
            Es, posits = Es[:i+cont], posits[:i+cont]
            for j in range(0,cont):
                Es[:i+j] = Es[i]
                posits[i+j+1] = posits[i+j] + np.array([l*np.cos(theta),l*np.sin(theta)])
                l = dx*np.random.random_sample()
            break
        else:
            dtheta = chooseTrajectory(CDF(Es[i],anglularfineness))
            theta+=dtheta
            Es[i+1] = postComptonEnergy(Es[i],dtheta)
            posits[i+1] = posits[i]+np.array([l*np.cos(theta),l*np.sin(theta)]) #dtheta or theta?
    return Es, posits

def polarPlot(df,color = None):
    theta = np.linspace(0,2*np.pi,len(df))
    fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta,df,color = color)
    return fig

def coolPolarPlot(fineness):
    fig,ax = plt.subplots(subplot_kw={'projection': '3d'})
    energies = np.linspace(0,1000,fineness)
    theta = np.linspace(0,np.pi*2,fineness)
    cmap = sns.color_palette("viridis",int(3/2*fineness))
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    for i,e in enumerate(energies):
        pdf = PDF(e,fineness)
        x = pdf*np.cos(theta)
        y = pdf*np.sin(theta)
        E = 0*y+e
        ax.plot(x,y,x*0-e,color = cmap[i+30])
    plt.show()

def multiPolarPlot(fineness):
    theta = np.linspace(0,2*np.pi,1000)
    fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})
    energies = np.linspace(0,1000,fineness)
    cmap = sns.color_palette("rocket",int(5/4.8*fineness))
    cmap2 = sns.color_palette("rocket",as_cmap=True)
    for i,e in enumerate(energies):
        df = PDF(e,1000)
        ax.plot(theta,df,color = cmap[i],linewidth = 1)
    norm = c.Normalize(vmin=0, vmax=1000*5/4.8)
    sm = cm.ScalarMappable(cmap=cmap2, norm=norm)
    plt.colorbar(mappable=sm, ax=ax)
    plt.show()
start = time()
tot = np.array([[0,0]])
print(tot.shape)

@njit( cache = True)
def aggregatePhotoScatt(n,l):
    tot = np.array([[0.0,0.0]])
    for i in range(n):
        _, pos = monteCarloPhotoscattering(2000000,0,1,l,1000)
        tot = np.append(tot,pos,axis = 0)
        if i%(n//10) == 0:
            print("10%\done")
        #lt.plot(pos[:,0],pos[:,1])
    return tot
l = 1*10**-13 # x = l*(1.6*10**16)**!!
l = 100/3 # for z???

tot = aggregatePhotoScatt(10000,l)
plt.hexbin(tot[:,0],tot[:,1], gridsize=(91,73), bins = "log",extent = (-40*l,40*l,-17*l,17*l))
plt.show()
 