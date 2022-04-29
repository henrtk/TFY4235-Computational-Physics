from SpinSim import *

def plotQuivers(A,M,N, speed = 1):
    plottingPlane = (0,1,2)   # (0, 1, 2) corresponds to the coordinates (x, y, z) respectively
    fig, ax = plt.subplots(1)
    X = np.arange(M)
    Y = np.arange(N)
    X,Y = np.meshgrid(X,Y)

    Q = ax.quiver(X,Y,A[0,:,:,plottingPlane[0]],A[0,:,:,plottingPlane[1]], A[0,:,:,plottingPlane[2]])

    def update_quiver(i,Q,A):
        """
        updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """
        U = A[i*speed,:,:,plottingPlane[0]]
        V = A[i*speed,:,:,plottingPlane[1]]
        C =  A[i*speed,:,:,plottingPlane[2]]

        Q.set_UVC(U,V,C)
        return Q,
    log.    info("Creating animation")
    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, A),
                               interval=30, blit=False, repeat = True)
    plt.show()
