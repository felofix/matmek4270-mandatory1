import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:
    def __init__(self):
        self.L = 1

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        x = np.linspace(0, self.L, N + 1)
        y = np.linspace(0, self.L, N + 1)
        self.h = self.L/N
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij', sparse=False)

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')           # Creating a sparse diagonal matrix with a list of lists. 
        D[0, :4] = 2, -5, 4, -1                                               # Specifying boundariess with precision of n^4.
        D[-1, -4:] = -1, 4, -5, 2
        return D

    def create_w(self):
        """Return the dispersion coefficient"""
        self.w = sp.sqrt(self.kx**2 + self.ky**2) #c = 1

    def ue(self):
        """Return the exact standing wave"""
        return sp.sin(self.kx*x)*sp.sin(self.ky*y)*sp.cos(self.w*t)

    def initialize(self):
        """Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        ue = self.ue()
        self.Unm1 = sp.lambdify((x, y, t), ue)(self.xij, self.yij, 0)
        self.Un = self.Unm1 + 0.5*(self.dt)**2*(self.D @ self.Unm1 + self.Unm1 @ self.D.T)

    def get_dt(self):
        """Return the time step"""
        dt = (self.cfl*self.h)
        return dt

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        uj = sp.lambdify((x, y, t), u)(self.xij, self.yij, t0)
        return np.sqrt(np.sum((self.Un-uj)**2))*self.h

    def apply_bcs(self):
        # Set boundary conditions
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.cfl = cfl
        self.create_mesh(N)
        self.kx, self.ky = mx*sp.pi, my*sp.pi
        
        self.create_w()
        self.dt = self.get_dt()
        self.D = self.D2(N)/(self.h**2)
        self.initialize()

        plotdata = {0: self.Unm1.copy()}
        
        if store_data == 1:
            plotdata[1] = self.Un.copy()

        for n in range(1, Nt):
            self.Unp1 = 2*self.Un - self.Unm1 + (self.dt)**2*(self.D @ self.Un + self.Un @ self.D.T)

            # Set boundary conditions
            self.apply_bcs()

            # Swap solutions
            self.Unm1 = self.Un
            self.Un = self.Unp1

            if n % store_data == 0:
                plotdata[n] = self.Unm1.copy() # Unm1 is now swapped to Un

        if store_data == -1:
            return self.h, self.l2_error(self.ue(), (Nt)*self.dt)
        if store_data > 0:
            return self.xij, self.yij, plotdata
        

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')           # Creating a sparse diagonal matrix with a list of lists. 
        D[0, :2] = -2, 2                                                       # Specifying boundariess with precision of n^4.
        D[-1, -2:] = 2, -2
        return D
        
    def ue(self):
        return sp.cos(self.kx*x)*sp.cos(self.ky*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    solN = Wave2D_Neumann()
    N0 = 2**(12)
    Nt = 10
    dx, err_dir = sol(N0, Nt, cfl=1/np.sqrt(2), mx=2, my=2, store_data=-1)
    dx, err_neu = solN(N0, Nt, cfl=1/np.sqrt(2), mx=2, my=2, store_data=-1)
    
    assert err_dir < 1e-15
    assert err_neu < 1e-15