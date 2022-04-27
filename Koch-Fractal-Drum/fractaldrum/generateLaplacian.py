from time import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba import prange
import scipy.sparse as spsp
from collections import deque
import logging
from joblib import dump
## GLOBALS
# max coord is 2147483648
dtypeK = np.int64 

## END GLOBALS
@jit(nopython = True, cache=True)
def is_point_in_path(x: dtypeK, y: dtypeK, kochCorner) -> np.bool_:
    # Determine if the point is in the polygon.
    #
    # params:
    #   x          : int        -> The x coordinates of point.
    #   y          : int        -> y coordinates of point.
    #   kochCorner : np.ndarray -> array of corner coordinates  [(x, y), (x, y), ...]
    #
    # Returns:
    #   True if the point is in the path or is a corner or on the boundary
    num = len(kochCorner)
    j = num - 1
    c = False
    for i in range(num):
        if (x == kochCorner[i,0]) and (y == kochCorner[i,1]):
            # point is a corner, we treat corners as outside, as u = 0 there
            return False
        if ((kochCorner[i,1] > y) != (kochCorner[j,1] > y)):
            slope = (x-kochCorner[i,0])*(kochCorner[j,1]-kochCorner[i,1])-(kochCorner[j,0]-kochCorner[i,0])*(y-kochCorner[i,1])
            #if slope == 0: Currently obsolete.
                # point is on boundary (and not a top-boundary.)
            #    return True
            if (slope < 0) != (kochCorner[j,1] < kochCorner[i,1]):
                c = not c
        j = i
    return c


def _tridiagLaplace(N,dtype=np.float64):
    """
    _protected_
    Generates tridiag-block for 2d laplacian
    """
    diagonal = [-1,4,-1]
    return spsp.diags(diagonal,[-1,0,1],shape=(N,N),dtype=dtype)


def nullify(x,y,twoDlaplacian):
    """
    Set the elements in the 2Dlaplacian relating to the point x, y to 0   
    """
    N = int(np.sqrt(twoDlaplacian.get_shape()[0]))
    points = _matrixpospoint(y,x,N)
    for i in points:
        twoDlaplacian[i[0],i[1]] = 0
    return  

@jit(nopython = True, cache = True)
def _matrixpospoint(i,j,N):
    """
    matrix notation, i is basicly x, j is basicly y
    currently not in use
    Note: possible that indexation sets inconsequential elements to zero
    """
    center = N*j+i 
    if j !=0 and j!=N-1:
        result = [(center  ,center), #point in question
                  (center  ,center+1), # v fixing right-left-neighbours
                  (center  ,center-1),
                  (center+1,center),
                  (center-1,center),
                  (center  ,center+N), # v fixing up-down-neighbours
                  (center  ,center-N),
                  (center+N,center),
                  (center-N,center)]
    elif (j == N-1):
        result = [(center  ,center),
                  (center  ,center-1),
                  (center-1,center),
                  (center  ,center-N),
                  (center-N,center)]
    else:
        result = [(center  ,center),
                  (center  , center+1),
                  (center+1,center),
                  (center  ,center+N),
                  (N+center,center)]
    return result

def laplacian2d(N,format,dtype = np.float64):
    """
    Generates a boundary condition less 2d laplacian
    using a five point central difference approximation  
    """

    innerDiag = _tridiagLaplace(N,dtype=dtype)
    outerDiag = -spsp.eye(N,dtype = dtype) 
    A = spsp.bmat([
                    [innerDiag if (i == j) 
                    else outerDiag if abs(i-j)==1
                    else None for i in range(N)]
                    for j in range(N)],
                    format=format)
    logging.info(f"Generated matrix size {A.get_shape()[0]}x{A.get_shape()[0]}, type: {type(A)}")
    return A

@jit(nopython = True, cache = True)
def shiftxY(arr : np.array,length):
    for i in range(len(arr)):
        arr[i][0] = arr[i][0] + length
        arr[i][1] = arr[i][1] + length
    return arr
        
def unpackeigvec(flatvector: np.ndarray,shape : tuple):
    flatvector.reshape((shape))
    return flatvector.reshape((shape))

def kochLaplacianBoundIs0(kochLen,kochCorners)->spsp.spmatrix:
    A = laplacian2d(kochLen,"dok")
    N = int(np.sqrt(A.get_shape()[0]))
    print("Iterating over all points")
    for x in prange(kochLen):
        for y in prange(kochLen):
            if not is_point_in_path(x,y,kochCorners):      
                points = _matrixpospoint(y,x,N)
                for i in points:
                    A[i[0],i[1]] = 0
    return A

def kochLaplacianBoundIs0v2(kochLen,kochCorners,dtype = np.float64)-> tuple(spsp.spmatrix,np.ndarray):
    A = laplacian2d(kochLen,"csc",dtype=dtype)
    logging.info("Iterating over all points")
    eliminateOutpoints(kochLen,kochCorners,A.data,A.indptr)
    logging.info("Converting sparse format")
    A = A.tocsr()
    logging.info("Half # of points generated")
    eliminateOutpoints(kochLen,kochCorners,A.data,A.indptr)
    # Should be csr on exit, faster iteration
    A.eliminate_zeros()
    indices = A.getnnz(0)>0
    # sort and eliminate diagonal zeroes.
    logging.info("Manipulating array")
    A = A[A.getnnz(1)>0][:,A.getnnz(0)>0]
    return A, indices


@jit(nopython = True, cache = True, parallel = True)
def eliminateOutpoints(kochLen, kochCorners, csr_data, csr_indptr):
    """
    Eliminates points thay are outside the
    """
    for x in prange(kochLen):
        if (x % kochLen//10 == 0):
            print("10% of points checked")
        for y in range(kochLen):
            if not is_point_in_path(x, y, kochCorners):      
                 csr_data[csr_indptr[kochLen*y + x] : csr_indptr[kochLen*y + x + 1]] = 0
    return

@jit(nopython=True, cache = True, parallel = True)
def generatePosits(kochlen,kochCorners):
    """
    Generates bool gridmap of koch fractal, parallel friendly
    True indicates grid point is inside of fractal

    params:
        kochlen     : int        -> length of grid axes for the specific koch fractal
        kochCorners : np.ndarray -> koch fractal corner coordinates 
    
    """
    gridmap = np.zeros(shape=(kochlen,kochlen),dtype = np.bool8)
    num = len(kochCorners)
    for x in prange(kochlen):
        for y in range(kochlen):
            j = num-1
            c = False
            for i in range(num):
                if (x == kochCorners[i,0]) and (y == kochCorners[i,1]):
                    # point is a corner, we treat corners as outside for now, as u = 0 there
                    break
                if ((kochCorners[i,1] > y) != (kochCorners[j,1] > y)):
                    slope = (x-kochCorners[i,0])*(kochCorners[j,1]-kochCorners[i,1])-(kochCorners[j,0]-kochCorners[i,0])*(y-kochCorners[i,1])
                    #if slope == 0: Quarantined due to being obsolete with current implementation
                        # point is on boundary
                    #    gridmap[y,x] = True
                    #    break
                    if (slope < 0) != (kochCorners[j,1] < kochCorners[i,1]):
                        c = not c
                j = i
            else:
                gridmap[y,x] = c
    return gridmap


def generateLaplace(posits) -> tuple:
    #SPECIAL CASE FOR  lowest and highest points, they only generate 1 below (first) and 1 above
    inGrid = np.argwhere(posits==1)
    A= spsp.eye(len(inGrid),format = "dok")*4
    que = deque([])
    print("Generating laplacian")
    start = time()
    for ind,(i,j) in enumerate(inGrid):
        #place = (posits[i+1,j] == 2)
        #A[ind,ind+1] = -1*place
        #A[ind+1,ind] = -1*place
        # alternatively:
        if (posits[i,j+1]):
            A[ind,ind+1] = -1
            A[ind+1,ind] = -1
        
        if len(que) > 0:

            if j==que[0][1]:
                lastcoord = que.popleft()[2]
                A[lastcoord,ind] = -1
                A[ind,lastcoord] = -1

        if (posits[i+1,j]):
            que.append((i,j,ind))
    print("done:",time()-start)
    return A.tocsr(),inGrid

# May be jittable some day.
def generateLaplace2(posits : np.ndarray) -> tuple:
    inGrid = np.argwhere(posits)
    A= spsp.eye(len(inGrid),format = "dok")*4
    que = np.empty(shape = (len(inGrid),3),dtype = np.int16)
    print(f"Generating sparse laplacian {len(inGrid)} x {len(inGrid)} ")
    queLast = 0
    queFirst = 0
    for ind,(i,j) in enumerate(inGrid):
        #place = (posits[i+1,j] == 2)
        #A[ind,ind+1] = -1*place
        #A[ind+1,ind] = -1*place
        # alternatively:
        if (posits[i,j+1]):
            A[ind,ind+1] = -1
            A[ind+1,ind] = -1
        
        # May want to just check if quelast is bigger than 0 but this may suffice with numba
        if queLast>queFirst:
            # check if queued coordinate has identical j coordinate 
            if j==que[queFirst][1]:
                lastcoord = que[queFirst][2]
                A[lastcoord,ind] = -1
                A[ind,lastcoord] = -1
                queFirst +=1
        if (posits[i+1,j]):
            que[queLast] = np.array([i,j,ind])
            queLast +=1
    
    return A.tocsr(),inGrid

def recreate(A,kochlen):
    recreation = np.zeros(shape = (kochlen**2,2), dtype = np.int32)
    for i in range(kochlen**2):
        if A[i,i] == 0: 
            recreation[i] = np.array([i//kochlen,i%kochlen]) #possible switchup of 
        #x, y seems to be swapped but that makes very little difference.
    plt.scatter(recreation[:,0],recreation[:,1], s = 0.1)
    plt.show()

def unpackVec(vec,grid,lenght):
    """
    "Unpack" eigenvector, that is, fold it back such that vector index correspond to 
    correct position within the fractal drum for plotting.
    
    params:
        vec   : np.ndarray                           -> vector to unpack
        grid  : np.ndarray(shape = (length, length)) -> grid to unpack to    
        length: int                                  -> length of grid
    """
    newVec = np.zeros((lenght,lenght))
    for i,j in zip(vec,grid):
        newVec[j] = i
    return newVec

#next up, fix eigvals!
if __name__ == "__main__":
    #plt.matshow(A.toarray()-C.toarray())
    import kochSubstrate
    a = kochSubstrate.KochSubstrate(4,5)
    A, Ind = kochLaplacianBoundIs0v2(a.len,a.kochCorners)
    np.save("l4,f5",A)
    np.save("Indsl4f5",Ind)

