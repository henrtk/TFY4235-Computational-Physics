import numpy as np
import logging
import matplotlib.pyplot as plt
from numba import prange, njit
dtypePref = np.int64

class KochSubstrate:
    initiator = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]],dtype=dtypePref)
    def __init__(self,l,fineness) -> None:
        self.l = l
        self.fineness = fineness
        self.grid, self.len, self.shift = gridpts(l,fineness)
        
        # generate the reference length between points
        self.h = 1/(4**(l)*fineness)
        logging.info(f"Initiate fractal corner generation in a {self.len}x{self.len} grid...")
        self.kochCorners = generateFractal(self.initiator,l,fineness)
        self.kochCorners = self.findBoundary()
        self.kochCorners = shiftxY(self.kochCorners, self.shift)
        print("Fractal corners done!")

    def populate(self):
        for i in range(self.len):
            for j in range(self.len):
                self.grid[i,j] = isPointInPoly(i,j,self.kochCorners)
        for i in self.findBoundary():
            if self.grid[i[0],i[1]]:
                self.grid[i[0],i[1]] = np.bool_(False) 

    def findBoundary(self):
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


def generateFractal(array,k,fineness):
    """
    Recursively generate the Koch-fractal's corners.

    params:
        array : np.ndarray -> The line segments to apply the fractal generation to
        
    """
    if k <= 0:
        return array
    else:
        # Somehow this expression for the lenght of the generated corners array is perfect. 
        # Found this by finding the relationship between the length of the array, the fineness, and
        # the resulting extra points at the tail of the corners. wtf  
        generated = np.zeros(shape=(8*(len(array)-5)*fineness+32+1,2), dtype = dtypePref)
        for i in range(len(array)-1):
            v1 = (array[i+1] - array[i])*fineness
            # 90 degree rotation
            v2 = np.array([v1[1],-v1[0]])
            p2 = array[i]*4*fineness + v1
            p3 = p2 + v2
            p4 = p3 + v1
            p7 = array[i+1]*4*fineness - v1 
            p6 = p7 - v2
            p5 = p6 - v1
            pmid=array[i]*4*fineness+2*v1
            generated[8*i:8*i+8] = np.array([array[i]*4*fineness,p2,p3,p4,pmid,p5,p6,p7],dtype=dtypePref)
        return generateFractal(generated,k-1,1)

@njit(cache = True)
def shiftxY(arr : np.array, length):
    for i in range(len(arr)):
        arr[i][0] = arr[i][0] + length
        arr[i][1] = arr[i][1] + length
    return arr

def gridpts(l : int, fineness : int, generateGrid = False):
    """
    Generate grid (parameters) where corners are exactly on a grid point with the smallest possible grid
    that fits all grid points 
    
    params:
        l        : int -> times the fractal pattern has been generated
        fineness : int -> amount of points between every corner point
    
    returns:
        x        : np.ndarray -> actual grid of x, y positions
        axes len : int        -> number of grid points on each axis
        extra    : int        -> additional lenght needed for grid (outside of initial square) to cover all corners 
    """
    length = 4**(l)*fineness
    extra = sum([4**(l-j) for j in range(1,l+1)])*fineness
    if generateGrid:
        x = np.zeros(shape = (2*extra+length+1,2*extra+length+1), dtype = np.bool_)
    else:
        x = np.zeros(shape =(2,2))
    return x, 2*extra+length+1, extra

@njit(cache = True)
def isPointInPoly(x: dtypePref, y: dtypePref, kochCorner) -> np.bool_:
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

