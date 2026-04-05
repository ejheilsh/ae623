import numpy as np 
from dataclasses import dataclass, field

@dataclass
class Uutil():
    """
    A wrapper class that returns primitive variables and other useful quantities from conservative state vector of the compressible Euler equations  
    Inputs:
    --------
    U: np.ndarray
        Conservative form of state vector for the compressible Euler equations 
    gamma: float
        Ratio of specific heats
    """
    U: np.ndarray   
    gamma: float    
    
    def __post_init__(self):
        self.rho = self.U[0]
        self.rhou = self.U[1]
        self.rhov = self.U[2]
        self.rhoE = self.U[3]
        
        # primitive state variables
        self.u = self.rhou/self.rho
        self.v = self.rhov/self.rho
        self.E = self.rhoE/self.rho
        
        # useful quantities
        self.vvec = np.array([self.u, self.v])  # velocity vector
        self.qsq = self.u**2 + self.v**2        # velocity magnitude squared q^2
        self.q = np.sqrt(self.qsq)              # velocity magnitude, q

        self.p = (self.gamma - 1) * (self.rhoE - 0.5 * self.rho * self.qsq) # pressure
        self.H = self.E + self.p / self.rho                                 # enthalpy
        self.c = np.sqrt(self.gamma * self.p / self.rho)                    # speed of sound
        self.M = self.q / self.c                                            # Mach number
    
        # if self.stdout==True:
        #     self.assert_messages()
            
    def assert_messages(self):
        """
        print statements when get rho < 0, p < 0, E < 0, sqrt of negative, divide by zero errors
        These can happen during startup transients, large CFL..
        Should NOT silently guard inside Uutil, as this will mask numerical instability
        let it fail in development, add assertions and fix root cause in the solver
        TODO move asserts after things get calculated in __post_init__
        """    
        if self.rho <= 0: 
            raise ValueError("Negative density detected")
        if self.p <= 0: 
            raise ValueError("Negative pressure detected")
        if self.E <= 0: 
            raise ValueError("Negative energy detected")
        
# NOTE OR CAN CREATE INIDIVUDAL HELPER FUNCTIONS TAKING IN THE STATE AND RETURNING A CERTAIN VALUE.... LIKE OPERATOR OVERLOADING IN AE588...

# in a regular function, would have to figure out what to return where. 
def U_conservative_to_primitive(U: np.ndarray, gamma: float, returnlst = ['']):
    rho = U[0]
    rhou = U[1]
    rhov = U[2]
    rhoE = U[3]
    
    # return primitive quantities 
    u = rhou/rho 
    v = rhov/rho
    E = rhoE/rho 
    
    # return useful quantities (velocity vector, velocity magnitude squared, pressure, enthalpy)
    vvec = np.array([u, v])     # velocity vector
    qsq = u**2 + v**2           # magnitude velocity squared 
    q = np.sqrt(qsq)            # magnitude velocity 
    
    p = (gamma - 1) * (rhoE - 0.5 * rho * qsq) # pressure
    H = E + p / rho     # enthalpy
    c = np.sqrt(gamma * p / rho)    # speed of sound
    M = q / c   # Mach number
    
    return 

# NOTE have to troubleshoot this util because there need to be guards added in...div by zero, invalid values...
if __name__=="__main__":
    Uex = np.array([
        1, 
        8,
        10,
        150,
    ])
    
    gamma = 1.4

    Utest = Uutil(Uex, gamma) 
    print(Utest.p)
    print(Utest.rho)
    
