from SpinSim import *

def plotQuivers(evol,M,N, speed = 1):
    plottingPlane = (0,1,2)   # (0, 1, 2) corresponds to the coordinates (x, y, z) respectively
    fig, ax = plt.subplots(1)
    X = np.arange(M)
    Y = np.arange(N)
    X,Y = np.meshgrid(X,Y)

    Q = ax.quiver(X,Y,evol[0,:,:,plottingPlane[0]],evol[0,:,:,plottingPlane[1]], evol[0,:,:,plottingPlane[2]])

    def update_quiver(i,Q,evol):
        """
        updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """
        U = evol[i*speed,:,:,plottingPlane[0]]
        V = evol[i*speed,:,:,plottingPlane[1]]
        C =  evol[i*speed,:,:,plottingPlane[2]]

        Q.set_UVC(U,V,C)
        return Q,
    log.info("Creating animation")
    anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, evol),
                               interval=30, blit=False, repeat = True)
    plt.show()
