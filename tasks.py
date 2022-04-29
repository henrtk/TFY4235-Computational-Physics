import numpy as np
import SpinSim as s
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotQuiver

standardConsts = s.Consts(ALPHA=0.1,GAMMA=0.176,J=1,\
KBT=0,B = np.array([0,0,1]),d_z=0,magMom = 5.788*10**-2)

def simOneSpinA(dt,steps):

    #figure used simOneSpinA(0.01,10000)
    S1 = s.normalize([0.1,0,1])
    spinState = s.spinlattice(1,1,start = S1,random=False)
    consts = standardConsts
    consts.ALPHA = 0.0
    spinEvolution = s.HeunsMethod3dLattice(spinState,dt,steps,consts)
    
    ts = np.arange(steps+1)*dt
    fig,axes = plt.subplots(2,1,sharex=True)
    labels = ("x","y","z")

    def cos(ts,w):
        return 0.1*np.cos(ts*w)
    w, _ = curve_fit(cos,ts,spinEvolution[:,0,0,0],p0 = (0.3))

    for i,ax in enumerate(axes[:2]):
        ax.plot(ts,spinEvolution[:,0,0,i], label = f"Spin evolution $S_{labels[i]}$",linewidth = 2)
        ax.set_ylabel(f"$S_{labels[i]}$")
        ax.set_xlabel("Time")
        ax.plot(ts,0.1*np.sin(ts*w),linestyle = ":",label = "sin($\omega$ t)",linewidth = 2)
        ax.plot(ts,0.1*np.cos(ts*w),linestyle = ":",label = "cos($\omega$ t)",linewidth = 2, color = "r")
        ax.set_ylim([-0.18,0.12])
        ax.legend(ncol = 3, loc = "lower center")
    plt.show()

    return

def simOneSpinB(dt,steps,alpha):
    # Used to generate pic : simOneSpinB(0.01,10000,0.1)
    
    S1 = s.normalize([0.1,0,1])
    print(S1)
    spinState = s.spinlattice(1,1,start = S1,random=False)
    consts = standardConsts
    consts.ALPHA = alpha
    spinEvolution = s.HeunsMethod3dLattice(spinState,dt,steps,consts)
    
    ts = np.arange(steps+1)*dt
    fig, axes = plt.subplots(2,1,sharex=True)
    labels = ("x","y","z")

    def expCos(x,w,T):
        return S1[0]*np.exp(-x*T)*np.cos(w*x)
    (w,T),_ = curve_fit(expCos,ts,spinEvolution[:,0,0,0],p0=(0.3,0.3*alpha))

    #plotting
    for i,ax in enumerate(axes):
        ax.plot(ts,spinEvolution[:,0,0,(4*i)//2], label = f"Spin evolution $S_{labels[(4*i)//2]}$",linewidth = 2)
        ax.set_ylabel(f"$S_{labels[(i*4)//2]}$")
        ax.set_xlabel("Time")
    #axes[0].plot(ts,expCos(ts,w,T),linestyle = ":",label = "exponential damping fit",linewidth = 2)
    axes[0].plot(ts,expCos(ts,w,alpha*w),linestyle = ":",label = "Exponential damping using $\\tau = \\frac{1}{\\alpha \\omega}$",linewidth = 2, color = "r")
    axes[0].legend()
    print("Damping half life difference:",T-consts.ALPHA*w)
    plt.show()
    return

#simOneSpinB(0.01,10000,0.1)

def simAtomicChain(dt,steps, atoms = 40):
    #simAtomicChain(0.001,40000, atoms = 40) 
    spinInitial = s.spinlattice(atoms,1,start=np.array([0,0,1]))
    
    standardConsts.ALPHA = 0.00

    spinInitial[0,0] = s.normalize(np.array([0.1,0.0,1]))
    spinEvolution = s.HeunsMethod3dLattice(spinInitial,dt,steps,standardConsts)
    fig,axes = plt.subplots(1,3, sharey=True)
    axes[0].set_ylabel("Time [2fs]")
    labels = ("x","y","z")
    for i,ax in enumerate(axes):
        ax.set_title(f"S$_{labels[i]}$")
        im = ax.imshow(spinEvolution[::2,:,0,i],aspect = "auto",cmap = "viridis")
        plt.colorbar(im, ax = ax)
        ax.set_xlabel("Atom position")
    
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax)
    plt.show()
    ts = np.arange(steps+1)*dt
    fig, axes = plt.subplots(1,3)
    for i in range(5):
        axes[0].plot(ts,spinEvolution[:,i,0,0],label = f"S$_x$ atom {i+1}")
        axes[1].plot(ts,spinEvolution[:,i,0,1],label = f"S$_y$ atom {i+1}")
        axes[2].plot(ts,spinEvolution[:,i,0,2],label = f"S$_z$ atom {i+1}")

    for i, ax in enumerate(axes):
        ax.set_title(f"$S_{labels[i]}$")
        ax.set_xlabel("Time [ps]")
        ax.legend()
    
    plt.show()
    
simAtomicChain(0.0001,400000,atoms = 100)  

