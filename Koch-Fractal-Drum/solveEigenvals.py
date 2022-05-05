import fractaldrum as util
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spspl
from joblib import load,dump
import time as t
import cupyx.scipy.sparse.linalg as cpspl
import cupyx.scipy.sparse as cpsp
from cupy import array as cparray
import logging

logging.basicConfig(level = logging.INFO,
                    format='%(asctime)s %(message)s')


def solveEigenValAssignment(level,fineness,numEigVects, save=False, returnEigenvectors = False):
    logging.info(f"Generating fractal l = {level}, fineness = {fineness}")
    fractal = util.KochSubstrate(level,fineness)
    logging.info("Fractal generated. Generating")
    laplacian, indices = util.kochLaplacianBoundIs0v2(fractal.len,fractal.kochCorners)
    logging.info("Initiating ARPACK call")
    
    eig_vals_vecs = spspl.eigsh(laplacian,numEigVects,sigma = 0, return_eigenvectors= returnEigenvectors)
    logging.info("ARPACK done")
    if save:
        np.save(f"Koch-Fractal-Drum\fractaldrum\__pycache__\l={level},f={fineness}",eig_vals_vecs)
    return eig_vals_vecs, indices, fractal.len, fractal.kochCorners, fractal.h

def unpackEigVec(vec,indices,gridlength):
    grid = np.zeros(gridlength**2)
    grid[indices] = vec
    grid = np.reshape(grid,(gridlength,gridlength))
    return grid


def plotEigvalsNaively(level,fineness,numEigVects):
    (vals, vecs), inds, lent,corns =solveEigenValAssignment(level,fineness,numEigVects=numEigVects)
    x, y  = np.arange(lent),np.arange(lent)
    for i in vecs.T:
        A = unpackEigVec(i,inds,lent)
        plt.pcolormesh(x,y,A,cmap="viridis_r",shading = "gouraud")
        plt.plot(corns[:,0],corns[:,1])
        plt.show()


def iDOS(w,eigvals):
    """
    Gives the amount of eigenvalues below a given omega
    Beware: units chosen such that v = 1 (as eigenvalues are prop to 
    w^2/v^2)

    params:
        eigvals : np.ndarray -> array-like of eigenvalues
        w       : float      -> upper limit
    
    returns:
        lenght  : int        -> # of eigenvalues below w 
    """
    
    s = w.copy()
    for i in range(len(w)):
        s[i] = len(eigvals[eigvals <= w[i]**2])
    
    return s

def deltaDOS(eigvals, h, plot = False):
    topLim     =    max(np.sqrt(eigvals)/h)
    waxis      =    np.linspace(0, topLim,1000)
    Area       =    1


    dDOS       =    Area/(4*np.pi)*waxis**2 - iDOS(waxis,eigvals/h**2)
    #print(iDOS(waxis,eigvals/h**2))

    from scipy.optimize import curve_fit 
    def _wToTheD(w, M, d):
        return M*w**d
    print(params[1])
    params, covs = curve_fit(_wToTheD,waxis,dDOS,p0=(0.01,1.5))
    
    plt.plot(waxis,dDOS)
    plt.plot(waxis,_wToTheD(waxis,params[0],params[1]))
    plt.show()
    return params[0], params[1]

def checkWeylConj(level, fineness):
    eig_vals_vecs, _,_,_, h = solveEigenValAssignment(level,fineness,1200,False,False)
    print(deltaDOS(eig_vals_vecs,h,True))
    return


def checkfromSavedLapl(filenameL,filenameI=0,k = 50):
    indices, Laplacian = 0,0
    with open(filenameL,"rb") as f:
        Laplacian = np.load(f, allow_pickle = True)
        print(type(Laplacian.item()))
        print(np.array2string(Laplacian,formatter={'float' : lambda x: "%.2f" % x}))
    eigvalsAndVec = spspl.eigsh(Laplacian.item(), k=k,sigma = 0)
    #with open(filenameI,"rb") as f2:
    #    indices = np.load(f2)      
    #np.save(filenameL+"eigvecs",eigvalsAndVec[1])
    #np.save(filenameL+"eigvals",eigvalsAndVec[0])
    

