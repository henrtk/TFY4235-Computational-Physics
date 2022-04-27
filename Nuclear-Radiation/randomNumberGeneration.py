import numpy as np
import matplotlib.pyplot as plt
from scipy import rand

# Quick explanation: This file is only meant to demonstrate 
# and visualize two common simple random number generation algorithms
# and hopefully visualize their flaws

def visualizeRNG(RNG,N,seed,histType = plt.hexbin):
    randNums = RNG(N**2, seed = seed)
    randnums = randNums.reshape((N,N))
    plt.matshow(randnums, cmap = "Greys")
    plt.show()
    return

def middleSquare(N : int,seed : int = 39840828890298010102920938450293) -> np.ndarray:
    """
    Middle square method for generating uniformally distributed random numbers.

    Never use this in a serious context.

    "Anyone who considers arithmetical methods of producing random digits is, of course, in a state of sin" (Von Neumann, 1949)

    Is it even correct?

    params: 
        N    : int ->  The number of random numbers to generate. 
        Seed : int ->  The starting point for generation of random numbers
        max  : int ->  which integer corresponds to the number 1.

    return: 
        random numbers : np.ndarray(N) -> Numpy array of N random uniformally distributed floats.
    """
    x = np.empty(N)
    generator = seed
    for i in range(N):
        length = len(str(generator)) - len(str(generator))%2
        randnum = generator*generator
        if length:
            generator = int((length%2*"0"+str(randnum))[length//2:length+length//2])
            x[i] = generator
        else:
            x[i:] = 0
            break
    return x

def linearCongruential(N, seed,a = 102, c = 6269, mod = 7703):
    x = np.empty(N+1)
    x[0] = seed
    for i in range(N):
        x[i+1] = (x[i]*a + c)%mod
    return x[1:]

