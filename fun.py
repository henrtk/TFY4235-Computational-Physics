import numpy as np
import matplotlib.pyplot as plt
import numba as nb

@nb.njit()
def generateAs(pct,mtpc, N : int):
    studs = np.zeros(N)
    for j in range(N):
        for i in range(mtpc):
            if np.random.uniform(0,1) < pct:
                studs[j] += 2
            else:
                studs[j] -= 2/3
        for i in range(20-mtpc):
            if np.random.uniform(0,1) < pct:
                studs[j] +=2
    return studs/40*100

M = 11
andel=np.zeros(M)
for j in range(0,20,2):
    for i in range(M):
        start = 0.89
        studs = generateAs(start+i/100,j,2_000_000)
        stats = np.histogram(studs,density=True,bins = [69,74,79,84,89,94,100])
        andel[i] = sum(stats[0][-2:])/sum(stats[0])
    plt.plot(np.linspace(0.89,1,M),andel)
plt.show()

