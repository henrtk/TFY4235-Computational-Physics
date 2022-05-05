import logging
import numpy as np
import spinSim as s
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import plotQuiver


standardConsts = s.Consts(
    ALPHA                   = 0.05,
    GAMMA                   = 0.176,
    J                       = 1,
    T                       = 0,
    B                       = np.array([0,0,1]),
    d_z                     = 0,
    magMom                  = 5.788*10**-2
    )

def simOneSpinA(dt,steps):

    #figure used simOneSpinA(0.01,10000)

    # Set up initial conditions 
    S1 = s.normalize([0.1,0,1])
    spinState = s.spinlattice(1,1,start = S1,random=False)
    consts = standardConsts
    consts.ALPHA = 0.0
    spinEvolution = s.HeunsMethodLattice(spinState,dt,steps,consts)
    
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

    spinState = s.spinlattice(1,1,start = S1,random=False)
    consts = standardConsts
    consts.ALPHA = alpha
    spinEvolution = s.HeunsMethodLattice(spinState,dt,steps,consts)
    
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


def simAtomicChain(dt,steps, atoms = 40, periodic = False):
    #simAtomicChain(0.001,40000, atoms = 40) 

    spinInitial          =   s.spinlattice(atoms,1,start=np.array([0,0,1]))

    standardConsts.ALPHA =   0.00 #0.08

    spinInitial[0,0]     =   s.normalize(np.array([0.1,0.0,1]))
    spinEvolution        =   s.HeunsMethodLattice(spinInitial,dt,steps,standardConsts,periodic)
    
    # the rest is plotting and plot setups. Boring!
    fig,axes             =   plt.subplots(1,3, sharey=True)
    labels               =   ("x", "y", "z")

    axes[0].set_ylabel("Time [2fs]")

    for i,ax in enumerate(axes):
        
        im = ax.imshow(spinEvolution[::2,:,0,i],aspect = "auto",cmap = "viridis")
        plt.colorbar(im, ax = ax)
        ax.set_xlabel("Atom position")
        ax.set_title(f"S$_{labels[i]}$")
    
    plt.show()

    # more plotting
    ts = np.arange(steps+1)*dt
    fig, axes = plt.subplots(1,3)
    for i in range(4):
        axes[0].plot(ts,spinEvolution[:,i,0,0],label = f"S$_x$ atom {i+1}")
        axes[1].plot(ts,spinEvolution[:,i,0,1],label = f"S$_y$ atom {i+1}")
        axes[2].plot(ts,spinEvolution[:,i,0,2],label = f"S$_z$ atom {i+1}")

    #axes[0].plot(ts,spinEvolution[:,40,0,0],label = f"S$_x$ atom {50}")
    #axes[1].plot(ts,spinEvolution[:,40,0,1],label = f"S$_y$ atom {50}")
    #axes[2].plot(ts,spinEvolution[:,40,0,2],label = f"S$_z$ atom {50}")

    for i, ax in enumerate(axes):
        ax.set_title(f"$S_{labels[i]}$")
        ax.set_xlabel("Time [ps]")
        ax.legend()

    # very cool quiver plot of spin states. Not suited to view with 0,0,1 init vectors! 
    # Arrow sizes are decided by first frame :(
    #plotQuiver.plotQuivers(spinEvolution,atoms,1)
    plt.show()


def sim1dGroundState(dt, steps, atoms, periodic, d_z, C : s.Consts, antiferro):
    C.B        *=   3
    C.d_z       =   d_z

    if antiferro:   
        C.J     =   -1

    spinInitial =   s.spinlattice(atoms,1,random=True)
    spinEvol    =   s.HeunsMethodLattice(spinInitial,dt,steps,C,periodic)

    fig,axes    =   plt.subplots(1,3, sharey=True)
    labels      =   ("x","y","z")

    axes[0].set_ylabel("Time [fs]")

    for i,ax in enumerate(axes):
        ax.set_title(f"$S_{labels[i]}$")
        ax.set_xlabel("Atom position")
        im = ax.imshow(spinEvol[:,:,0,i],aspect = "auto",cmap = "viridis")
        plt.colorbar(im, ax = ax)
    
    plt.show()    
    return
    

def simGroundstate2d(dt : float, steps : int, atomsX : int, atomsY : int, C: s.Consts):
    # Initialize 
    spinInitial =   s.spinlattice(atomsY,atomsX,random = True)
    spinEvol    =   s.HeunsMethodForgetful(spinInitial,dt,steps,C,periodic=True)
    
    fig,axes    =   plt.subplots(1,3, sharey=True)
    labels      =   ("x","y","z")

    axes[0].set_ylabel("Time [2fs]")

    for i, ax in enumerate(axes):
        ax.set_title(f"$S_{labels[i]}$")
        ax.set_xlabel("Atom position")
        ax.set_ylabel("Atom position")
        im = ax.imshow(spinEvol[:,:,i], aspect = "auto", cmap = "viridis")
        plt.colorbar(im, ax = ax)
    plt.show()
    return 


def findMagnetizationOverTime(dt,steps,atomsX,atomsY,T, C : s.Consts):
    """
    Generates time evolution of magnetization over time in a temperature influenced
    plane of atoms with spin, for a given temperature T. Void, generates a plot.
    Intial condition is all states equal to [0, 0, 1]. Plots the average of the 
    equilibrium magnetization as a horizontal line.

    params:
        dt          :    float   -> time step size
        steps       :    int     -> # of iterations for Heun's method
        atoms(X,Y)  :    int     -> # of atoms along the X, Y direction in the atomic plane
        T           :    float   -> Temperature to run simulation for
        C           :    Consts  -> class carrying physical parameters

    returns:
        void
    """
    Ms = np.empty(steps)

    C.KBT = T*0.0862  # kb in meV
    
    lattice = s.spinlattice(atomsX,atomsY,start=np.array([0,0,1.0]))
    interval = 50
    
    logging.disable("INFO")
    
    for i in tqdm(range(steps)):
        lattice      =       s.HeunsMethodForgetful(lattice,dt,interval,C,True)
        Ms[i],_      =       s.magnetizationTimeavg(lattice)
    
    plt.plot(np.arange(steps)*dt*interval,Ms, label = "$\langle M(t)\\rangle$, T = 10 [K]", linewidth = 1)
    plt.xlabel("Time [ps]")
    avgMagEquil = np.mean(Ms[:-steps//3])
    plt.hlines(avgMagEquil,xmin = 0, xmax = interval*steps*dt, 
               label = "Equilibrium magnetization", linestyle = "--", color = "r")
    
    # re initialize with new temp 1K
    lattice = s.spinlattice(atomsX,atomsY,start=np.array([0,0,1.0]))
    C.KBT = 0.0862
    for i in tqdm(range(steps)):
        lattice      =       s.HeunsMethodForgetful(lattice,dt,interval,C,True)
        Ms[i],_      =       s.magnetizationTimeavg(lattice)
    
    plt.plot(np.arange(steps)*dt*interval,Ms, label = "$\langle M(t)\\rangle$, T = 1 [K]", linewidth = 1)
    avgMagEquil = np.mean(Ms[:-steps//3])
    plt.hlines(avgMagEquil,xmin = 0, xmax = interval*steps*dt, 
               label = "Equilibrium magnetization", linestyle = "--", color = "m",linewidth = 2)
    plt.legend(loc = "lower left",bbox_to_anchor=(0.45,0.45))
    plt.show()
    return  


def curieSweep(dT,stepsT,C : s.Consts, interval:int = 1_000,atoms = 30,tag = "", plotLive=False):
    """
    Calculate and plot equilibrium magnetization vs temperature.
    First initiate all spins in lattice in the z-direction and choose first T = dT
    
    Then evolve spin state dt*stepsPerIter and calculate and save average magnetization for the final state.
    Repeat (iters) times.
    The system should now be approximately in equilibrium. Evolve another 500 steps and calculate ensemble time
    average by averaging the averages of the magnetization every tenth time step. Save this ensemble average.

    increment T by dT,
    repeat this stepsT times.

    params:
        dT          : float     -> temperature step for the sweep
        stepsT      : int       -> amount of times to increment by dT
        C           : s.Consts  -> class holding all relevant physical parameters, jit-friendly
        interval    : int       -> # of time steps to calculate equilibrium avg magnetization over
        atoms       : int       -> the # of atoms on each axis in the lattice, total atoms x atoms spin states
        tag         : str       -> extra tag for the filename to save the generated data to.
        plotLive    : bool      -> decide if we want the plot to be generated as-we-go, to visualize progression
    
    returns:
        plot of equilibrium magnetization for every T, 
        plot of magnetization evolution (to visualize reaching equilibrium)
    """
    # set up variables, mostly for data visualization
    stepsPerIter    =    160
    iters           =    40
    dt              =    0.001    
    ts              =    np.arange(iters)*dt*stepsPerIter
    Ts              =    np.logspace(0,2.85,stepsT+1, base=5)
    magnetVals      =    np.empty((stepsT+1,iters))
    magnetAvg       =    np.empty(stepsT+1)
    magnetStds      =    np.empty(stepsT+1)
    kb              =    0.0862             # [meV/T]


    fig, axes = plt.subplots(2,1)
   
    logging.disable("INFO")
    for i,T in tqdm(enumerate(Ts)):
        C.KBT                           =       kb*T 
        forw                            =       s.spinlattice(atoms,atoms,start=[0,0,1])
        axes[0].clear()
        axes[1].clear()
        for j in range(iters):
            forw                        =       s.HeunsMethodForgetful(forw,dt,stepsPerIter,C,True)
            magnetVals[i,j], _          =       s.magnetizationTimeavg(forw)
        
        equilibrium                     =       s.HeunsMethodLattice(forw,dt,interval,C,True)
        magnetAvg[i], magnetStds[i]     =       s.magnetizationTimeavg(equilibrium, separation = 1)
        
        if plotLive:
            if i >=2:    
                axes[0].pcolormesh(ts,Ts[:i],magnetVals[:i],vmin=-0.05, vmax=1, shading="auto")

            #plot progress, averages.
            axes[1].plot(Ts[:i], magnetAvg[:i])
            #Plot 95% confidence intervals, 1.95 is close enough to 2.
            axes[1].fill_between(Ts[:i], magnetAvg[:i] + 2*magnetStds[:i], y2=magnetAvg[:i]-2*magnetStds[:i], alpha=0.5)
            plt.pause(0.5)

    im = axes[0].pcolormesh(ts,Ts,magnetVals,vmin=-0.05, vmax=1,shading="auto")
    axes[0].set_title("$\langle M_z(T,t)\\rangle$")
    axes[0].set_xlabel("Time [ps]")
    axes[0].set_ylabel("Temp. [K]")

    axes[1].plot(Ts, magnetAvg)
    axes[1].fill_between(Ts, magnetAvg + 2*magnetStds, y2=magnetAvg-2*magnetStds, alpha=0.5)
    axes[1].set_xlabel("T [K]")
    axes[1].set_ylabel("$M_z(T)$ equilibrium")

    plt.colorbar(im, ax=axes[0])
    #plt.plot(Ts,magnetAvg)
    #plt.fill_between(Ts, magnetAvg + magnetStds, y2=magnetAvg-magnetStds, alpha=0.5)
    #plt.ylim([-.1,1.1])
    #plt.title("Equilibrium Magnetization as a function of T")
    #plt.xlabel("T [K]")
    #plt.ylabel("Ensemble magnetization")
    
    
    np.save(f"MagnetAvg {atoms}, dT = {dT}, maxT = {dT*stepsT}, {tag}", magnetAvg, allow_pickle = True)
    np.save(f"MagnetStd {atoms}, dT = {dT}, maxT = {dT*stepsT}, {tag}", magnetStds, allow_pickle = True)


def _plotSavedData():
    import os
    Ts = np.linspace(1,41,41)
    for avg, std in zip(os.scandir("magavg2"),os.scandir("magstd2")):
        if "tonight" in str(avg):
            Ts = np.logspace(0,2.85,101,base=5)
        
            #print(str(avg))
            #Ts = np.linspace(1,40,41)
            magnetAvg = np.load(avg,allow_pickle=True)
            magnetStds = np.load(std,allow_pickle=True)
            plt.title("Phase diagrams for different field strengths.")
            plt.xlabel("Temperature [K]", fontsize= 12)
            plt.plot(Ts[:-10], magnetAvg[:-10], label = str(avg)[-20:-14]+"T")
            plt.fill_between(Ts[:-10], magnetAvg[:-10] + 2*magnetStds[:-10], y2=magnetAvg[:-10]-2*magnetStds[:-10], alpha=0.4)
    plt.legend(fontsize = 12)
    plt.show()
    pass


"""
# Generate many of the figures used in the text

#simOneSpinB(0.01,10000,0.1)

#simAtomicChain(0.001,20_000,40,True)

#sim1dGroundState(0.001,100_000, atoms = 100, periodic=True,d_z =0.1,  C = standardConsts,antiferro=False)#imAtomicChain(0.003,50000,atoms = 100, periodic=True)  


consts2d = s.Consts(
    ALPHA       =       0.3,
    GAMMA       =       0.176,
    J           =       1,
    T           =       1,
    B           =       np.array([0,0,1.72]),
    d_z         =       0,
    magMom      =       5.788*10**-2
    )   

#findMagnetizationOverTime(0.001,3_000,20,20,T=10,C=consts2d)


consts2dsweep = s.Consts(
    ALPHA       =       0.3,
    GAMMA       =       0.176,
    J           =       1,
    T           =       1,
    B           =       np.array([0.0,0.0,10]),
    d_z         =       0,
    magMom      =       5.788*10**-2
    )    


#for i in (1,5,10,30):
#    
#    consts2d.B = np.array([0.0,0.0,i])
#    curieSweep(1, 100, consts2dsweep, interval =  10_000, atoms = 40, tag = f"B = {i} tonight")

#plt.plot()
"""