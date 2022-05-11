import numba as fast
from numba import float64

ConstTypes = [("ALPHA" , float64),("GAMMA" , float64),("J",float64),
("KBT", float64),("B", float64[:]),("d_z" , float64),("magMom",float64)]

@fast.experimental.jitclass(ConstTypes)
class Consts(object):
    """
    Compact Numba-readable Costs object that carries the simulation constants.
    Time is given in pico seconds.

    params:
        ALPHA : Dampening constant 0 < ALPHA < 1 [unitless]
        GAMMA : Gyromagnetic constant       [1/(T ps)], ~ 0.178 
        J     : Coupling constant           [meV]       ~ 1
        KBT   : Thermal energy              [meV]       ~ 0.086 meV/K * T
        B     : Magnetic field              [T]         ~ 1.72 
        d_z   : Anisotropy constant         [meV]       ~ 0.1*J
        magMom: Magnetic moment             [meV/T]     ~ 5.788*10**-2
    """
    def __init__(self,ALPHA,GAMMA,J,T,B,d_z,magMom):
        self.ALPHA = ALPHA      
        self.GAMMA = GAMMA      
        self.J = J
        self.KBT = T*0.0862 
        self.B = B*1.0 
        self.d_z = d_z
        self.magMom = magMom

