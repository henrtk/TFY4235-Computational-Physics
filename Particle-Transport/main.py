import diffusion as d
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as clr


_,pos = d.simulateParticlesDetailed(20000,631400,5,10**-3)
d.histogramtransport(pos, fine = 10)

#colors = sns.color_palette("viridis",10)
#particleTrajs = [d.Particle(0,0,0.1+np.exp(i/2),0.0007) for i in range(10)]
#for i in range(10):
#    particleTrajs[i].update(4000000)
#    d.plotParticleTrajectory(particleTrajs[i].traj,particleTrajs[i].dT,period =particleTrajs[i].period, color = colors[i] )
#plt.colorbar(cm.ScalarMappable(norm =clr.LogNorm(0.1,np.exp(5)), cmap="viridis"))
#plt.show()