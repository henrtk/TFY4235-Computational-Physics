from time import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba import prange
import scipy.sparse as spsp
from collections import deque
## GLOBALS
# max coord is 2147483648
dtypeK = np.int64 

## END GLOBALS

class KochSubstrate:
    initiator = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]],dtype=dtypeK)
    def __init__(self,l,fineness,populate = False) -> None:
        self.l = l
        self.fineness = fineness
        self.grid, self.len, self.extra = gridpts(l,fineness,populate)
        print("Generating corners...")
        self.kochCorners = newForward(self.initiator,l,fineness)
        self.kochCorners = self.findBoundary()
        self.kochCorners = shiftxY(self.kochCorners, self.extra)
        print("Done!")
        if populate:
            self.populate()
    def populate(self):
        for i in range(self.len):
            for j in range(self.len):
                self.grid[i,j] = is_point_in_path(i,j,self.kochCorners)
        for i in self.findBoundary():
            if self.grid[i[0],i[1]]:
                self.grid[i[0],i[1]] = np.bool_(False) 
    def findBoundary(self):
        l = self.l
        kochCorner = self.kochCorners
        f = self.fineness
        if f == 1:
            return kochCorner
        boundary = np.zeros(shape=(f*len(kochCorner),2), dtype = np.int32) 
        for pos,vecs in enumerate(zip(kochCorner[1:],kochCorner[:-1])):
            for i in range(f):
                boundary[pos*f+i] = vecs[1] + i*(vecs[0]-vecs[1])//f
        return boundary

    def plotgrid(self):
        fig, ax = plt.subplots(1)
        for i in range(self.len):
            for j in range(self.len):
                if self.grid[i,j]:
                    ax.scatter(i,j)
        plt.show()
    def plotKoch(self,plotAlone = True):
        plt.plot(self.kochCorners[:,1],self.kochCorners[:,0],color = "red",linewidth =2)
        if plotAlone:
            plt.show()

#@jit(nopython=True) Deprecated
def corners(arr0 ,arr1):
    v2 = np.zeros(2, dtype=np.int32)
    v1 = (arr1-arr0)//4
    v2 = np.array([v1[1],-v1[0]],dtype=dtypeK)
    p2 = arr0 + v1
    p3 = p2 + v2
    p4 = p3 + v1
    p7 = arr1 - v1 
    p6 = p7 - v2
    p5 = p6 - v1
    pmid=arr0+2*v1
    result = np.array([arr0,p2,p3,p4,pmid,p5,p6,p7],dtype=dtypeK)
    return result

def newForward(array,k,fineness):
    if k <= 0:
        return array
    else:
        generated = np.zeros(shape=(8*(len(array)-5)*fineness+32+1,2),dtype = dtypeK)
        for i in range(len(array)-1):
            v1 = (array[i+1] - array[i])*fineness
            v2 = np.array([v1[1],-v1[0]])
            p2 = array[i]*4*fineness + v1
            p3 = p2 + v2
            p4 = p3 + v1
            p7 = array[i+1]*4*fineness - v1 
            p6 = p7 - v2
            p5 = p6 - v1
            pmid=array[i]*4*fineness+2*v1
            generated[8*i:8*i+8] = np.array([array[i]*4*fineness,p2,p3,p4,pmid,p5,p6,p7],dtype=dtypeK)
        return newForward(generated,k-1,1)

def stepFroward(array,k, fineness):
    if k == 0:
        return array[2:]
    else:
        a = np.ndarray((2,2),dtype=dtypeK)
        for i,j in zip(array[2:-1],array[3:]):
            a = np.append(a,corners(i*4*fineness,j*4*fineness),axis = 0)
        a = np.append(a,np.array([[0,0]], dtype=dtypeK),axis = 0)
        return stepFroward(a,k-1,1)

def gridpts(l : int, fineness : int, generateGrid = False):
    length = 4**(l)*fineness
    extra = sum([4**(l-j) for j in range(1,l+1)])*fineness
    if generateGrid:
        x = np.zeros(shape = (2*extra+length+1,2*extra+length+1), dtype = np.bool_)
    else:
        x = np.zeros(shape =(2,2))
    return x, 2*extra+length+1, extra


@jit(nopython=True)
def is_point_in_path(x: dtypeK, y: dtypeK, kochCorner) -> np.bool_:
    # Determine if the point is in the polygon.
    #
    # Args:
    #   x -- The x coordinates of point.
    #   y -- The y coordinates of point.
    #   kochCorner -- a list of tuples [(x, y), (x, y), ...]
    #
    # Returns:
    #   True if the point is in the path or is a corner or on the boundary
    num = len(kochCorner)
    j = num - 1
    c = False
    for i in range(num):
        if (x == kochCorner[i,0]) and (y == kochCorner[i,1]):
            # point is a corner, we treat corners as outside for now, as u = 0 there
            return False
        if ((kochCorner[i,1] > y) != (kochCorner[j,1] > y)):
            slope = (x-kochCorner[i,0])*(kochCorner[j,1]-kochCorner[i,1])-(kochCorner[j,0]-kochCorner[i,0])*(y-kochCorner[i,1])
            #if slope == 0: Currently obsolete.
                # point is on boundary and not a top-boundary.
            #    return True
            if (slope < 0) != (kochCorner[j,1] < kochCorner[i,1]):
                c = not c
        j = i
    return c

def _tridiagLaplace(N):
    """
    ___private___
    Generates tridiag-block for 2d laplacian
    """
    diagonal = [-1,4,-1]
    return spsp.diags(diagonal,[-1,0,1],shape=(N,N),dtype=np.float64)


def nullify(x,y,twoDlaplacian):
    """
    Set the elements in the 2Dlaplacian relating to the point x,y to 0   
    """
    N = int(np.sqrt(twoDlaplacian.get_shape()[0]))
    points = _matrixpospoint(y,x,N)
    for i in points:
        twoDlaplacian[i[0],i[1]] = 0
    return  

@jit(nopython = True)
def _matrixpospoint(i,j,N):
    """
    matrix notation, i is basicly x, j is basicly y
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

def laplacian2d(N,format):
        innerDiag = _tridiagLaplace(N)
        outerDiag = -spsp.eye(N,dtype = np.float64) 
        A = spsp.bmat([[innerDiag if (i == j) else outerDiag if abs(i-j)==1
                else None for i in range(N)]
                for j in range(N)], format=format)
        print("Size: ",str(A.get_shape()[0])+"x"+str(A.get_shape()[0]),"type: ",type(A))
        return A

@jit(nopython = True)
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

def kochLaplacianBoundIs0v2(kochLen,kochCorners)->spsp.spmatrix:
    A = laplacian2d(kochLen,"csr")
    N = int(np.sqrt(A.get_shape()[0]))
    print("Iterating over all points")
    findallinRow(kochLen,kochCorners,A.data,A.indptr)
    #A.tocsc() #Should be csr on exit, faster iteration
    #findallinCol(kochLen,kochCorners,A.data,A.indptr)
    A.eliminate_zeros()
    return A
@jit(nopython = True, parallel = True)
def findallinRow(kochLen,kochCorners,csr_data,csr_indptr):
    for x in prange(kochLen):
        for y in prange(kochLen):
            if not is_point_in_path(x,y,kochCorners):      
                 csr_data[csr_indptr[kochLen*x+y]:csr_indptr[kochLen*x+y+1]] = 0
    return
@jit(nopython = True, parallel = True)
def findallinCol(kochLen,kochCorners,csc_data,csc_indptr):
    for x in prange(kochLen):
        for y in prange(kochLen):
            if not is_point_in_path(x,y,kochCorners):      
                 csc_data[csc_indptr[kochLen*x+y]:csc_indptr[kochLen*x+y+1]] = 0
    return
@jit(nopython=True, parallel = True)
def generatePosits(kochlen,kochCorner):
    """
    int(kochlen) -> length of axis 
    Generates gridmap of koch, parallel friendly
    """
    gridmap = np.zeros(shape=(kochlen,kochlen),dtype = np.int8)
    num = len(kochCorner)
    for x in prange(kochlen):
        for y in prange(kochlen):
            j = num-1
            c = False
            for i in range(num):
                if (x == kochCorner[i,0]) and (y == kochCorner[i,1]):
            # point is a corner, we treat corners as outside for now, as u = 0 there
                    gridmap[y,x] = 2
                    break
                if ((kochCorner[i,1] > y) != (kochCorner[j,1] > y)):
                    slope = (x-kochCorner[i,0])*(kochCorner[j,1]-kochCorner[i,1])-(kochCorner[j,0]-kochCorner[i,0])*(y-kochCorner[i,1])
                    #if slope == 0: Quaranntined due to being obsolete with current implementation
                        # point is on boundary
                    #    gridmap[y,x] = True
                    #    break
                    if (slope < 0) != (kochCorner[j,1] < kochCorner[i,1]):
                        c = not c
                j = i
            else:
                gridmap[y,x] = c
    return gridmap

def generateLapl(posits) -> tuple:
    #SPECIAL CASE FOR  lowest and highest points, they only generate 1 below (first) and 1 above
    inGrid = np.argwhere(posits==1)
    A= spsp.eye(len(inGrid),format = "dok")*4
    que = deque([])
    print("starting")
    start = time()
    for ind,(i,j) in enumerate(inGrid):
        #place = (posits[i+1,j] == 2)
        #A[ind,ind+1] = -1*place
        #A[ind+1,ind] = -1*place
        # alternatively:
        if (posits[i,j+1] == 1):
            A[ind,ind+1] = -1
            A[ind+1,ind] = -1
        
        if len(que) > 0:

            if j==que[0][1]:
                lastcoord = que.popleft()[2]
                A[lastcoord,ind] = -1
                A[ind,lastcoord] = -1

        if (posits[i+1,j] == 1):
            posits[i,j]
            que.append((i,j,ind))
    print("done:",time()-start)
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
    newVec = np.zeros((lenght,lenght))
    for i,j in zip(vec,grid):
        newVec[j] = i
    return newVec

    return
#next up, fix eigvals!
if __name__ == "__main__":

    a = KochSubstrate(3,2)
    b = generatePosits(a.len,a.kochCorners)
    B = generateLapl(b)
    #A = kochLaplacianBoundIs0(a.len,a.kochCorners)
    #C = kochLaplacianBoundIs0v2(a.len,a.kochCorners) ## Error in this func! fuck.
    #plt.matshow(A.toarray())
    plt.matshow(b)
    plt.matshow(B.toarray())
    #plt.matshow(A.toarray()-C.toarray())
    plt.show()