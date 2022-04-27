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

logging.basicConfig(level = logging.info)


def solveEigenValAssignment(level,fineness,numEigVects):
    logging.info(f"Generating fractal l = {level}, fineness = {fineness}")
    fractal = util.kochSubstrate(level,fineness)
    logging.info("Fractal generated. Generating")
    laplacian, indices = util.kochLaplacianBoundIs0v2(fractal.len,fractal.fineness)
    eigvals,eigvects = spspl.eigsh(laplacian,100,sigma = 0)







# RUN AND SAVE DOES NOT WORK RIGHT NOW, VERSION CLASH!
def runAndSave(level,fineness,eigenvects, plot = 0,save = True, CuDa = False,savefigs=False):
    koch = util.KochSubstrate(level,fineness)
    print("generating A")
    start = t.time()
    posits = util.generatePosits(koch.len,koch.kochCorners)
    B,grid = util.generateLaplace2(posits) 
    del posits
    end = t.time()
    print("A generated, time for completion:",end-start,"\nCalculating eigs")
    #B = util.laplacian2d(koch.len,"csc")
    #B = util.kochLaplacianBoundIs0v2(koch.len,koch.kochCorners)
    #plt.matshow(A.toarray())
    #plt.matshow(B.toarray())
    if CuDa:
        #size = int(np.sqrt(B.shape[0]))
        B = cpsp.csr_matrix(B)
        eigvalsAndVec = cpspl.eigsh(B, k=eigenvects)
        #x,y = np.meshgrid(np.arange(size),np.arange(size))
        #V = np.exp(-6*(x-size/2)**2/size-6*(y-size/2)**2/size)/5
        #V = np.reshape(V,(size**2))
        #V = cparray(V)
    #Remember to normalize and such. 
    #else:
    else:
        eigvalsAndVec = spspl.eigsh(B, k=eigenvects,which ="LM",sigma = 0)
    print("Done! This took:", t.time()-start,"seconds")
    if save:
        with open("__pycache__/Eigenvalues for l = "+str(level) + " fine = "+str(fineness)+" k = "+str(eigenvects),"wb") as f:
            print("dumping")
            start = t.time()
            if CuDa:
                eigvalsAndVec = eigvalsAndVec.get()
            dump(eigvalsAndVec,f,3)
            print("Dump complete, time: ",t.time()-start)
    if plot == 0:   
        for i in range(len(eigvalsAndVec[1][:])-1):
            vec  = util.unpackVec(eigvalsAndVec[1][:,i],grid,koch.len)
            I = vec
            #I = np.reshape(eigvalsAndVec[1][:,i],(koch.len,koch.len)).real
            fig,ax = plt.subplots(1)
            if CuDa:
                I = I.get()
            ax.imshow(I, cmap="viridis",)
            ax.plot(koch.kochCorners[:,0],koch.kochCorners[:,1],color = "red",linewidth=0.5)
            if savefigs:
                fig.savefig("./figures/l"+str(level)+"f"+str(fineness)+"k"+str(i)+"CuDA="+str(CuDa))
                plt.close()
            else:
                plt.show()
    if plot == 3:
        x,y =np.meshgrid(np.arange(koch.len),np.arange(koch.len))
        for i in range(len(eigvalsAndVec[1][:])-1):
            vec  = util.unpackVec(eigvalsAndVec[1][:,i],grid,koch.len)
            I = vec
            #I = np.abs((np.reshape((eigvalsAndVec[1][:,i]),(koch.len,koch.len),"C")))**2
            fig,ax = plt.subplots(subplot_kw={"projection" : "3d"})
            print(eigvalsAndVec[0][i])
            if CuDa:
                I = I.get()
            try:
                ax.plot_surface(x,y,I, cmap="viridis")
            except: 
                print(I,"error")
            #ax.plot3D(koch.kochCorners[:,1],koch.kochCorners[:,0],koch.kochCorners[:,0]*0,color = "red",linewidth =0.5, zorder = 3)
            plt.show()        

def readAndPlot(filename,plottype):
    with open(filename,"rb") as f:
        eigvalsAndVecs = load(f)
        length = int(np.sqrt(len(eigvalsAndVecs[1][:,0])))
    for i in range(len(eigvalsAndVecs[1][:])):
        veigs=np.reshape(eigvalsAndVecs[1][:,i],(length,length))
        I = np.abs(veigs)**2
        fig,ax = plt.subplots(subplot_kw={"projection" : "3d"})
        x,y =np.meshgrid(np.arange(length),np.arange(length))
        ax.plot_surface(x,y,I, cmap="viridis",)
        #ax.plot3D(koch.kochCorners[:,1],koch.kochCorners[:,0],koch.kochCorners[:,0]*0,color = "red",linewidth =0.5, zorder = 3)
        plt.show()        
    return
#runAndSave(3,5,20,save = False, plot = 3, savefigs=False , CuDa=False)
#runAndSave(1,2,40,plot=False,s
# ave=False,CuDa=False)
#readAndPlot("Eigenvalues for l = 5 fine = 3 k = 20",0)

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
    
checkfromSavedLapl("l5,f3.npy")
