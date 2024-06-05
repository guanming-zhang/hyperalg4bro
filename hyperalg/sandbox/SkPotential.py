import numpy as np
import finufft
class Sk_potential:
    def __init__(self, *args):
        self.Kvec = args[0]
        self.Vk = args[1]
        self.L = args[2]
        self.Sk0 = args[3]
        self.ndim = len(self.L)
        Kmag = np.sqrt(self.Kvec**2 + (self.Kvec**2).T)
        self.tol = 0.1+0.1*Kmag/48
        assert len(self.L) == len(self.Sk0.shape) == len(self.Vk.shape), 'incompatible dimensionality in L, Sk0, Vk'

    def get_energy(self,r):
        LL = np.array(np.reshape(self.L,(1,-1)))
        rr = np.array(r).reshape((-1, len(self.L)))
        assert np.isfinite(rr).all(),'xy is not finite'
        rr -= np.round(rr/LL)*LL
        rr *= 2*np.pi/LL
        assert np.isfinite(rr).all(),'xy is not finite after put in box'
        assert np.amax(rr)<=np.pi and np.amin(rr) > -np.pi
        N = rr.shape[0]
        c = np.ones(N)+0j
        Nk = self.Kvec.shape[-1]
        if self.ndim == 2:
            rho = finufft.nufft2d1(rr[:,0],rr[:,1],c,n_modes = (Nk,Nk))
            #rho[int(Nk/2),int(Nk/2)] *= 0 
        elif self.ndim == 3:
            rho = finufft.nufft3d1(rr[:,0],rr[:,1],rr[:,2],c,n_modes = (Nk,Nk,Nk))
            #rho[int(Nk/2),int(Nk/2),int(Nk/2)] *= 0
        assert np.isfinite(rho).all(),'rho is not finite'
        Sk = (np.absolute(rho)**2)/N
        Skdiff = (Sk - self.Sk0)/np.where(self.Sk0==0, 1,self.Sk0)
        Skdiff2 = Skdiff*Skdiff
        w = 1/(1+np.exp(100*(self.tol-Skdiff2)))
        dw = 100*w*(1-w)
        phi_step = np.sum(self.Vk*Skdiff2*w)
        return phi_step
    
    def get_energy_gradient(self,r):
        LL = np.array(np.reshape(self.L,(1,-1)))
        rr = np.array(r).reshape((-1, len(self.L)))
        assert np.isfinite(rr).all(),'xy is not finite'
        rr -= np.round(rr/LL)*LL
        rr *= 2*np.pi/LL
        assert np.isfinite(rr).all(),'xy is not finite after put in box'
        assert np.amax(rr)<=np.pi and np.amin(rr) > -np.pi
        N = rr.shape[0]
        c = np.ones(N)+0j
        Nk = self.Kvec.shape[-1]
        if self.ndim == 2:
            rho = finufft.nufft2d1(rr[:,0],rr[:,1],c,n_modes = (Nk,Nk))
