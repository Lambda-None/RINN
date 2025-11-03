import gc
import torch
import numpy as np
from torchmin import least_squares

####################### 1D Poisson Equations ####################################
class Poisson1DBase:
    def __init__(self, domain, Reg=[1,1]):
        self.domain = domain
        self.device = domain.device
        self.Reg = Reg

    def assemble_basis_terms(self, model, X_ini, X_bcs, X_res):
        H_bcs = model.get_last_hidden_layer(X_bcs)
        _, H_xx = model.compute_gradients(X_res)
        f = self.source(X_res)
        g = self.BC_fun(X_bcs)
        return H_bcs, H_xx, f, g

    def auto_solver(self, model, X_ini, X_bcs, X_res, coef=None):
        H_bcs, H_xx, f, g = self.assemble_basis_terms(model, X_ini, X_bcs, X_res)
        
        A = torch.cat([-H_xx*self.Reg[0], H_bcs*self.Reg[1]], dim=0)
        B = torch.cat([f*self.Reg[0], g*self.Reg[1]], dim=0)

        coef_sol = torch.linalg.lstsq(A, B).solution

        loss_equ = torch.sqrt(torch.mean(torch.square(f - (-H_xx@coef_sol))))
        loss_bcs = torch.sqrt(torch.mean(torch.square(g - H_bcs@coef_sol)))
        loss = loss_equ*self.Reg[0] + loss_bcs*self.Reg[1]
        return coef_sol, loss

    def BC_fun(self, x):
        return self.exact(x)

    def exact(self, x):
        raise NotImplementedError

    def source(self, x):
        raise NotImplementedError

class Poisson1D_1(Poisson1DBase):
    def __init__(self, domain, Reg=[1,1]):
        super().__init__(domain, Reg)

    def exact(self, x):
        u = torch.sin(2 * torch.pi * x) * torch.cos(3 * torch.pi * x)
        return u.to(self.device)

    def source(self, x):
        f = 13 * torch.pi**2 * torch.sin(2 * torch.pi * x) * torch.cos(3 * torch.pi * x) \
            + 12 * torch.pi**2 * torch.cos(2 * torch.pi * x) * torch.sin(3 * torch.pi * x)
        return f.to(self.device)

class Poisson1D_k(Poisson1DBase):
    def __init__(self, domain, k, Reg=[1,1]):
        super().__init__(domain, Reg)
        self.k = k

    def exact(self, x):
        u = torch.sin(self.k * torch.pi * x)
        return u.to(self.device)

    def source(self, x):
        f = self.k**2 * torch.pi**2 * torch.sin(self.k * torch.pi * x)
        return f.to(self.device)


####################### 2D Poisson Equations ####################################
class Poisson2DBase:
    def __init__(self, domain, Reg=[1,1]):
        self.domain = domain
        self.device = domain.device
        self.Reg = Reg

    def assemble_basis_terms(self, model, X_ini, X_bcs, X_res):
        H_bcs = model.get_last_hidden_layer(X_bcs)
        _, _, H_xx, H_yy = model.compute_gradients(X_res)
        f = self.source(X_res)
        g = self.BC_fun(X_bcs)
        return H_xx, H_yy, H_bcs, f, g

    def auto_solver(self, model, X_ini, X_bcs, X_res, coef=None):
        H_xx, H_yy, H_bcs, f, g = self.assemble_basis_terms(model, X_ini, X_bcs, X_res)
        Laplace = (H_xx + H_yy) 

        A = torch.cat([-(H_xx + H_yy)*self.Reg[0], H_bcs*self.Reg[1]], dim=0)
        B = torch.cat([f*self.Reg[0], g*self.Reg[1]], dim=0)

        coef_sol = torch.linalg.lstsq(A, B).solution

        loss_equ = torch.sqrt(torch.mean(torch.square(f - (-Laplace@coef_sol))))
        loss_bcs = torch.sqrt(torch.mean(torch.square(g - H_bcs@coef_sol)))
        loss = loss_equ*self.Reg[0] + loss_bcs*self.Reg[1]
        return coef_sol, loss

    def exact(self, x):
        raise NotImplementedError

    def source(self, x):
        raise NotImplementedError

    def BC_fun(self, x):
        return self.exact(x)
    
class Poisson2D_1(Poisson2DBase):
    def __init__(self, domain, Reg=[1,1]):
        super().__init__(domain, Reg)

    def exact(self, X):
        pi = torch.pi
        x = X[:, 0:1]
        y = X[:, 1:2]
        u = - (2 * torch.cos((3 / 2) * pi * x + 2 * pi / 5) + (3 / 2) * torch.cos(3 * pi * x - pi / 5)) * \
                (2 * torch.cos((3 / 2) * pi * y + 2 * pi / 5) + (3 / 2) * torch.cos(3 * pi * y - pi / 5))
        return u.to(self.device)

    def source(self, X_res):
        pi = torch.pi
        x = X_res[:, 0:1]
        y = X_res[:, 1:2]

        A_x = 2 * torch.cos((3 / 2) * pi * x + 2 * pi / 5) + (3 / 2) * torch.cos(3 * pi * x - pi / 5)
        B_y = 2 * torch.cos((3 / 2) * pi * y + 2 * pi / 5) + (3 / 2) * torch.cos(3 * pi * y - pi / 5)
        term1_Axx = -2 * ((3 * pi / 2)**2) * torch.cos((3 / 2) * pi * x + 2 * pi / 5)
        term2_Axx = -(3 / 2) * (3 * pi)**2 * torch.cos(3 * pi * x - pi / 5)
        A_xx = term1_Axx + term2_Axx
        term1_Byy = -2 * ((3 * pi / 2)**2) * torch.cos((3 / 2) * pi * y + 2 * pi / 5)
        term2_Byy = -(3 / 2) * (3 * pi)**2 * torch.cos(3 * pi * y - pi / 5)
        B_yy = term1_Byy + term2_Byy
        laplacian = (A_xx * B_y + A_x * B_yy)
        return laplacian.to(self.device)
    
class Poisson2D_k(Poisson2DBase):
    def __init__(self, domain, k, Reg=[1,1]):
        super().__init__(domain, Reg)
        self.k = k

    def exact(self, X):
        pi = torch.pi
        x = X[:, 0:1]
        y = X[:, 1:2]
        u = torch.sin(self.k * pi * x) * torch.sin(self.k * pi * y)
        return u.to(self.device)

    def source(self, X_res):
        pi = torch.pi
        x = X_res[:, 0:1]
        y = X_res[:, 1:2]

        f = (2 * self.k**2 * pi**2) * torch.sin(self.k * pi * x) * torch.sin(self.k * pi * y)
        return f.to(self.device)


####################### ND Poisson Equations ####################################
class PoissonNDBase:
    def __init__(self, domain):
        self.domain = domain
        self.device = domain.device

    def auto_solver(self, model, X_ini, X_bcs, X_res, coef=None):
        Delta_H = model.laplacian_last_hidden(X_res).detach()
        f = self.source(X_res)

        coef_sol = torch.linalg.lstsq(-Delta_H, f).solution

        Res = -Delta_H@coef_sol - f
        loss = torch.sqrt(torch.mean(torch.square(Res)))

        del Delta_H, f, Res
        torch.cuda.empty_cache()
        gc.collect()

        return coef_sol, loss

    def exact(self, x):
        raise NotImplementedError

    def source(self, x):
        raise NotImplementedError

    def BC_fun(self, x):
        return self.exact(x) 
    
class PoissonND(PoissonNDBase):
    def __init__(self, domain):
        super().__init__(domain)

    def exact(self, X):
        d = X.shape[1]
        C = (3/2)**d
        s = X.sum(dim=1, keepdim=True) / d            # m = (1/d) sum_i x_i
        A = s**2 + torch.sin(s)                       # A(m)
        P = torch.prod(1 - X**2, dim=1, keepdim=True) # prod_i (1 - x_i^2)
        return P * A * C

    def source(self, X):
        d = X.shape[1]
        C = (3/2)**d
        s = X.sum(dim=1, keepdim=True) / d
        A = s**2 + torch.sin(s)
        A1 = 2*s + torch.cos(s)                       # dA/dm
        A2 = 2 - torch.sin(s)                         # d^2A/dm^2 = 2 - sin(m)
        P = torch.prod(1 - X**2, dim=1, keepdim=True)

        x = X                                         # (N,d)
        q = 1 - x**2                                  # (N,d)
        h = -2 * x / q                                # (N,d) = ∂P/∂x_k / P
        B = h * A + (1.0 / d) * A1                    # (N,d)

        term1 = h * B                                 # (N,d)
        term2_p1 = -2 * (1 + x**2) / (q**2) * A       # (N,d)
        term2_p2 = h * (1.0 / d) * A1                 # (N,d)
        term2_p3 = (A2 / (d**2))                      # (N,1) -> broadcast to (N,d)
        term2 = term2_p1 + term2_p2 + term2_p3

        lap = P * (term1 + term2).sum(dim=1, keepdim=True)  # Δu
        f = -lap
        return C * f
    

####################### 1D Heat Equations ####################################
class Heat1DBase:
    def __init__(self, domain, Reg):
        self.domain = domain
        self.device = domain.device
        self.Reg = Reg

    def assemble_basis_terms(self, model, X_ini, X_bcs, X_res):
        H_ics = model.get_last_hidden_layer(X_ini)
        H_bcs = model.get_last_hidden_layer(X_bcs)
        _, H_t, H_xx, _ = model.compute_gradients(X_res)
        f = self.source(X_res)
        g = self.BC_fun(X_bcs)
        h = self.IC_fun(X_ini)
        return H_t, H_xx, H_bcs, H_ics, f, g, h
    
    def auto_solver(self, model, X_ini, X_bcs, X_res, coef=None):
        H_t, H_xx, H_bcs, H_ics, f, g, h = self.assemble_basis_terms(model, X_ini, X_bcs, X_res)

        A = torch.cat([(H_t-H_xx)*self.Reg[0], 
                       H_bcs*self.Reg[1], 
                       H_ics*self.Reg[2]], dim=0)
        B = torch.cat([f*self.Reg[0], g*self.Reg[1], h*self.Reg[2]], dim=0)

        coef_sol = torch.linalg.lstsq(A, B).solution

        loss_equ = torch.sqrt(torch.mean(torch.square(f - (H_t-H_xx)@coef_sol)))
        loss_bcs = torch.sqrt(torch.mean(torch.square(g - H_bcs@coef_sol)))
        loss_ics = torch.sqrt(torch.mean(torch.square(h - H_ics@coef_sol)))
        loss = self.Reg[0]*loss_equ + self.Reg[1]*loss_bcs + self.Reg[2]*loss_ics
        return coef_sol, loss
    
    def exact(self, X):
        raise NotImplementedError

    def source(self, X):
        raise NotImplementedError
    
    def IC_fun(self, X_ini):
        return self.exact(X_ini)

    def BC_fun(self, X_bcs):
        return self.exact(X_bcs)

class Heat1D_k(Heat1DBase):
    def __init__(self, domain, k, Reg=[1,1,1]):
        super().__init__(domain, Reg)
        self.k = k

    def exact(self, X):
        u = torch.exp(-X[:,1:2]) * torch.sin(self.k * torch.pi * X[:,0:1]) 
        return u.to(self.device)

    def source(self, X):
        f = (-1 + (self.k * torch.pi)**2) * torch.exp(-X[:,1:2]) * torch.sin(self.k * torch.pi * X[:,0:1]) 
        return f.to(self.device)
    

####################### 1D Advection Equations ####################################
class Advection1Dbase:
    def __init__(self, domain, c, Reg):
        self.c = c
        self.domain = domain
        self.device = domain.device
        self.Reg = Reg

    def assemble_basis_terms(self, model, X_ini, X_bcs, X_res):
        mask_1 = X_bcs[:, 0] == 1
        mask_2 = X_bcs[:, 0] == -1
        X_bc1 = X_bcs[mask_1]
        X_bc2 = X_bcs[mask_2]
        H_ics = model.get_last_hidden_layer(X_ini)
        H_bc1 = model.get_last_hidden_layer(X_bc1)
        H_bc2 = model.get_last_hidden_layer(X_bc2)
        H_x, H_t, _, _ = model.compute_gradients(X_res)
        f = self.source(X_res)
        g = self.BC_fun(X_bc1)
        h = self.IC_fun(X_ini)
        return H_x, H_t, H_ics, H_bc1, H_bc2, f, g, h
    
    def auto_solver(self, model, X_ini, X_bcs, X_res, coef=None):
        H_x, H_t, H_ics, H_bc1, H_bc2, f, g, h = self.assemble_basis_terms(model, X_ini, X_bcs, X_res)

        A = torch.cat([(H_t + self.c * H_x)*self.Reg[0], 
                       (H_bc1 - H_bc2)*self.Reg[1], 
                       (H_ics)*self.Reg[2]], dim=0)
        B = torch.cat([f*self.Reg[0], g*self.Reg[1], h*self.Reg[2]], dim=0)

        coef_sol = torch.linalg.lstsq(A, B).solution

        loss_equ = torch.sqrt(torch.mean(torch.square(f - (H_t@coef_sol + self.c * H_x@coef_sol))))
        loss_bcs = torch.sqrt(torch.mean(torch.square((H_bc1@coef_sol - H_bc2@coef_sol))))
        loss_ics = torch.sqrt(torch.mean(torch.square(h - H_ics@coef_sol)))
        loss = self.Reg[0] * loss_equ + self.Reg[1] * loss_bcs + self.Reg[2] * loss_ics
        return coef_sol, loss

    def exact(self, X):
        u = torch.zeros_like(X[:, 1:2])
        return u.to(self.device)

    def source(self, X):
        f = torch.zeros_like(X[:, 1:2])
        return f.to(self.device)

    def IC_fun(self, X):
        raise NotImplementedError

    def BC_fun(self, X):
        U_bcs = torch.zeros_like(X[:, 1:2])
        return U_bcs

class Advection_1(Advection1Dbase):
    def __init__(self, domain, c=0.4, Reg=[1,1,1]):
        super().__init__(domain, c, Reg)

    def IC_fun(self, X_ini):
        U_ini = torch.exp(0.5*torch.sin(2 * torch.pi * X_ini[:, 0:1]))-1
        return U_ini

class Advection_2(Advection1Dbase):
    def __init__(self, domain, c=0.4, Reg=[1,1,1]):
        super().__init__(domain, c, Reg)

    def IC_fun(self, X_ini):
        U_ini = torch.sin(2 * torch.pi * X_ini[:, 0:1]) * torch.cos(3 * torch.pi * X_ini[:, 0:1])
        return U_ini


####################### 1D Wave Equations ####################################
class Wave1Dbase:
    def __init__(self, domain, c, Reg):
        self.domain = domain
        self.device = domain.device
        self.c = c
        self.Reg = Reg

    def assemble_basis_terms(self, model, X_ini, X_bcs, X_res):
        H_bcs = model.get_last_hidden_layer(X_bcs)
        H_ics = model.get_last_hidden_layer(X_ini)
        _, _, H_xx, H_tt = model.compute_gradients(X_res)
        _, H_t_ics, _, _ = model.compute_gradients(X_ini)
        f = self.source(X_res)
        g = self.BC_fun(X_bcs)
        h = self.IC_fun(X_ini)
        ht = self.IC_grad(X_ini)
        return H_xx, H_tt, H_bcs, H_ics, H_t_ics, f, g, h, ht
    
    def auto_solver(self, model, X_ini, X_bcs, X_res, coef=None):
        H_xx, H_tt, H_bcs, H_ics, H_t_ics, f, g, h, ht = self.assemble_basis_terms(model, X_ini, X_bcs, X_res)

        A = torch.cat([
            (H_tt - (self.c ** 2) * H_xx)*self.Reg[0], 
            H_bcs*self.Reg[1], 
            H_ics*self.Reg[2], 
            H_t_ics*self.Reg[3]
            ], dim=0)
        B = torch.cat([
            f*self.Reg[0],
            g*self.Reg[1],
            h*self.Reg[2],
            ht*self.Reg[3]
            ], dim=0)
        
        coef_sol = torch.linalg.lstsq(A, B).solution

        loss_equ = torch.sqrt(torch.mean(torch.square(f - (H_tt - (self.c ** 2) * H_xx)@coef_sol)))
        loss_bcs = torch.sqrt(torch.mean(torch.square(g - H_bcs@coef_sol)))
        loss_ics = torch.sqrt(torch.mean(torch.square(h - H_ics@coef_sol)))
        loss_ics_t = torch.sqrt(torch.mean(torch.square(ht - H_t_ics@coef_sol)))
        loss = self.Reg[0]*loss_equ + self.Reg[1]*loss_bcs + self.Reg[2]*loss_ics + self.Reg[3]*loss_ics_t
        return coef_sol, loss

    def exact(self, X):
        u = torch.zeros_like(X[:, 0:1])
        return u.to(self.device)

    def source(self, X):
        f = torch.zeros_like(X[:, 0:1])
        return f.to(self.device)

    def BC_fun(self, X):
        U_bcs = torch.zeros_like(X[:, 1:2])
        return U_bcs
    
    def IC_fun(self, X):
        raise NotImplementedError
    
    def IC_grad(self, x):
        raise NotImplementedError


class Wave_1(Wave1Dbase):
    def __init__(self, domain, c, Reg=[1,1,1,1]):
        super().__init__(domain, c, Reg=[1,1,1,1])

    def IC_fun(self, X_ini):
        U_ini = torch.sin(1 * torch.pi * X_ini[:, 0:1]) \
              + torch.sin(2 * torch.pi * X_ini[:, 0:1]) \
              + torch.sin(3 * torch.pi * X_ini[:, 0:1])
        return U_ini

    def IC_grad(self, X_ini):
        U_ini_t = torch.zeros_like(X_ini[:, 0:1])
        return U_ini_t
    
class Wave_2(Wave1Dbase):
    def __init__(self, domain, c, Reg=[1,1,1,1]):
        super().__init__(domain, c, Reg)

    def IC_fun(self, X_ini):
        U_ini = torch.sin(2 * torch.pi * X_ini[:, 0:1]) \
              + torch.sin(4 * torch.pi * X_ini[:, 0:1]) 
        return U_ini

    def IC_grad(self, X_ini):
        U_ini_t = torch.zeros_like(X_ini[:, 0:1])
        return U_ini_t
    
class Wave_3(Wave1Dbase):
    def __init__(self, domain, c, Reg=[1,1,1,1]):
        super().__init__(domain, self.IC_fun, self.IC_grad, c, Reg)

    def IC_fun(self, X_ini):
        U_ini = torch.sin(1 * torch.pi * X_ini[:, 0:1]) \
              + torch.sin(3 * torch.pi * X_ini[:, 0:1]) 
        return U_ini

    def IC_grad(self, X_ini):
        U_ini_t = torch.zeros_like(X_ini[:, 0:1])
        return U_ini_t

####################### 1D Helmholtz Equations ####################################
class Helmholtz1Dbase:
    def __init__(self, domain, Lambda, Beta, Reg):
        self.domain = domain
        self.device = domain.device
        self.dtype = domain.dtype
        self.Lambda = Lambda  
        self.Beta = Beta 
        self.Reg = Reg
        self._precomputed = None  # 用于缓存预计算结果

    def precompute_basis_terms(self, model, X_ini, X_bcs, X_res):
        """预计算并缓存所有基础项，避免重复计算"""
        if self._precomputed is None:
            H = model.get_last_hidden_layer(X_res)
            H_bcs = model.get_last_hidden_layer(X_bcs)
            _, H_xx = model.compute_gradients(X_res)
            
            H = H.detach()
            H_bcs = H_bcs.detach()
            H_xx = H_xx.detach()
            
            f = self.source(X_res).flatten()
            g = self.BC_fun(X_bcs).flatten()
            
            self._precomputed = {
                'H': H,
                'laplacian': H_xx,
                'H_bcs': H_bcs,
                'f': f,
                'g': g
            }
        return self._precomputed
    
    def auto_solver(self, model, X_ini, X_bcs, X_res, coef=None):
        terms = self.precompute_basis_terms(model, X_ini, X_bcs, X_res)
        H = terms['H']
        laplacian = terms['laplacian']
        H_bcs = terms['H_bcs']
        f = terms['f']
        g = terms['g']
        
        def compute_residuals(coef):
            coef = coef.view(-1)
            u_res = H @ coef
            sin_u = torch.sin(u_res)
            
            Res_equ = laplacian @ coef - self.Lambda * u_res + self.Beta * sin_u - f
            Res_bcs = H_bcs @ coef - g
            
            return Res_equ, Res_bcs
        
        def nonlinear_system(coef):
            Res_equ, Res_bcs = compute_residuals(coef)
            Res = torch.cat([self.Reg[0] * Res_equ, self.Reg[1] * Res_bcs], dim=0)
            return Res
        
        def loss_compute(coef):
            Res_equ, Res_bcs = compute_residuals(coef)
            loss_equ = torch.sqrt(torch.mean(torch.square(Res_equ)))
            loss_bcs = torch.sqrt(torch.mean(torch.square(Res_bcs)))
            loss = self.Reg[0] * loss_equ + self.Reg[1] * loss_bcs
            return loss
        
        N = model.layers[-2]
        coef0 = torch.zeros((N,), device=self.device, dtype=self.dtype)
            
        result = least_squares(
            fun=nonlinear_system,
            x0=coef0,                     
            method='trf',                 
            ftol=1e-6,                   
            xtol=1e-6,
            gtol=1e-6,
            tr_solver='exact',            
            tr_options={'regularize': True},
            max_nfev=500,                 
            verbose=2
        )

        coef_opt = result.x  # (p,) tensor
        coef_sol = coef_opt.unsqueeze(-1)
        loss = loss_compute(coef_opt)
        
        self._precomputed = None

        del result
        torch.cuda.empty_cache()
        gc.collect()

        return coef_sol, loss

    def exact(self, x):
        raise NotImplementedError

    def source(self, X):
        raise NotImplementedError
    
    def BC_fun(self, X):
        raise NotImplementedError


class Helmholtz_1D(Helmholtz1Dbase):
    def __init__(self, domain, Lambda=5, Beta=1, Reg=[1,1]):
        super().__init__(domain, Lambda, Beta, Reg)
        self.Lambda = Lambda  
        self.Beta = Beta  
        
    def exact(self, X):
        u = torch.sin(3*torch.pi*X+3*torch.pi/20) * torch.cos(2*torch.pi*X-2*torch.pi/5) + 3/2 + X/10
        return u.to(self.device)
    
    def source(self, X):
        device = self.device

        # 定义角频率与相位
        a = 3.0 * torch.pi
        b = 2.0 * torch.pi
        a0 = 3.0 * torch.pi / 20.0
        b0 = -2.0 * torch.pi / 5.0

        # 基本三角项
        A = torch.sin(a * X + a0)
        B = torch.cos(b * X + b0)
        cosA = torch.cos(a * X + a0)
        sinB = torch.sin(b * X + b0)

        # 原函数 u
        u = A * B + 3.0/2.0 + X / 10.0

        u_xx = - (a**2 + b**2) * A * B  
        u_xx = - (a**2 + b**2) * A * B - 2.0 * a * b * cosA * sinB

        f = u_xx - self.Lambda * u + self.Beta * torch.sin(u)

        return f.to(device)
    
    def BC_fun(self, X_bcs):
        return self.exact(X_bcs)
    
####################### 2D Helmholtz Equations ####################################
class Helmholtz2Dbase:
    def __init__(self, domain, Lambda, Beta, Reg):
        self.domain = domain
        self.device = domain.device
        self.dtype = domain.dtype
        self.Lambda = Lambda  
        self.Beta = Beta 
        self.Reg = Reg
        self._precomputed = None  # 用于缓存预计算结果

    def precompute_basis_terms(self, model, X_ini, X_bcs, X_res):
        """预计算并缓存所有基础项，避免重复计算"""
        if self._precomputed is None:
            H = model.get_last_hidden_layer(X_res)
            H_bcs = model.get_last_hidden_layer(X_bcs)
            _, _, H_xx, H_yy = model.compute_gradients(X_res)
            
            H = H.detach()
            H_bcs = H_bcs.detach()
            H_xx = H_xx.detach()
            H_yy = H_yy.detach()
            
            f = self.source(X_res).flatten()
            g = self.BC_fun(X_bcs).flatten()
            
            laplacian = H_xx + H_yy
            
            self._precomputed = {
                'H': H,
                'laplacian': laplacian,
                'H_bcs': H_bcs,
                'f': f,
                'g': g
            }
        return self._precomputed
    
    def auto_solver(self, model, X_ini, X_bcs, X_res, coef=None):
        terms = self.precompute_basis_terms(model, X_ini, X_bcs, X_res)
        H = terms['H']
        laplacian = terms['laplacian']
        H_bcs = terms['H_bcs']
        f = terms['f']
        g = terms['g']
        
        def compute_residuals(coef):
            coef = coef.view(-1)
            u_res = H @ coef
            sin_u = torch.sin(u_res)
            
            Res_equ = laplacian @ coef - self.Lambda * u_res + self.Beta * sin_u - f
            Res_bcs = H_bcs @ coef - g
            
            return Res_equ, Res_bcs
        
        def nonlinear_system(coef):
            Res_equ, Res_bcs = compute_residuals(coef)
            Res = torch.cat([self.Reg[0] * Res_equ, self.Reg[1] * Res_bcs], dim=0)
            return Res
        
        def loss_compute(coef):
            Res_equ, Res_bcs = compute_residuals(coef)
            loss_equ = torch.sqrt(torch.mean(torch.square(Res_equ)))
            loss_bcs = torch.sqrt(torch.mean(torch.square(Res_bcs)))
            loss = self.Reg[0] * loss_equ + self.Reg[1] * loss_bcs
            return loss
        
        if coef is not None:
            coef0 = coef.squeeze(-1)
            perturb = torch.rand_like(coef0)
            coef0 = coef0 + 0.01*perturb
        else:
            N = model.layers[-2]
            coef0 = torch.zeros((N,), device=self.device, dtype=self.dtype)
            
        result = least_squares(
            fun=nonlinear_system,
            x0=coef0,                     
            method='trf',                 
            ftol=1e-6,                   
            xtol=1e-6,
            gtol=1e-6,
            tr_solver='exact',            
            tr_options={'regularize': True},
            max_nfev=150,                 
            verbose=2
        )

        coef_opt = result.x  # (p,) tensor
        coef_sol = coef_opt.unsqueeze(-1)
        loss = loss_compute(coef_opt)
        
        self._precomputed = None

        del result
        torch.cuda.empty_cache()
        gc.collect()

        return coef_sol, loss
    
    def exact(self, x):
        raise NotImplementedError

    def source(self, X):
        raise NotImplementedError
    
    def BC_fun(self, X):
        raise NotImplementedError

class Helmholtz_2D(Helmholtz2Dbase):
    def __init__(self, domain, Lambda=5, Beta=1, Reg=[1,1]):
        super().__init__(domain, Lambda, Beta, Reg)
        self.Lambda = Lambda  
        self.Beta = Beta  
        
    def exact(self, X):
        u = (torch.sin(3*torch.pi*X[:,0:1]+3*torch.pi/20) * torch.cos(2*torch.pi*X[:,0:1]-2*torch.pi/5) + 3/2 + X[:,0:1]/10) * \
            (torch.sin(2*torch.pi*X[:,1:2]+3*torch.pi/20) * torch.cos(3*torch.pi*X[:,1:2]-2*torch.pi/5) + 3/2 + X[:,1:2]/10)

        return u.to(self.device)
    
    def source(self, X):
        x, y = X[:, 0:1], X[:, 1:2]
        
        pi = torch.pi
        pi2 = pi ** 2
        alpha = 3.0 * pi / 20.0
        beta = -2.0 * pi / 5.0
        
        def compute_term(coord, a, b):
            s = torch.sin(a * pi * coord + alpha)
            c = torch.cos(b * pi * coord + beta)
            term_val = s * c + 1.5 + coord / 10.0
            
            term_dd = - (a**2 + b**2) * pi2 * (s * c) \
                    - 2 * a * b * pi2 * torch.cos(a * pi * coord + alpha) * torch.sin(b * pi * coord + beta)
            
            return term_val, term_dd
        
        A, A_dd = compute_term(x, 3.0, 2.0)
        
        B, B_dd = compute_term(y, 2.0, 3.0)
        
        u = A * B
        
        lap = A_dd * B + A * B_dd
        
        f = lap - self.Lambda * u + self.Beta * torch.sin(u)
        
        return f.to(self.device)
    
    def BC_fun(self, X_bcs):
        return self.exact(X_bcs)
    
 
####################### Paremetric Poisson Equation ####################################
class PoissonInverse:
    def __init__(self, domain):
        self.domain = domain
        self.device = domain.device
        self.dtype = domain.dtype

    def Nonlinear_system_torch(self, coef, H_xx, H_yy, H_bcs, H_data, f, g, u_data):
        alpha = coef[0]           # 标量
        beta = coef[1:]           # M*1向量

        r1 = H_xx @ beta + alpha* (H_yy @ beta) - f
        r2 = H_bcs @ beta - g
        r3 = H_data @ beta - u_data

        Residual = torch.cat([r1, r2, r3], dim=0)

        return Residual
    
    def auto_solver(self, model, X_res, X_bcs, X_data, u_data):
        _, _, H_xx, H_yy = model.compute_gradients(X_res)
        H_bcs = model.get_last_hidden_layer(X_bcs)
        H_data = model.get_last_hidden_layer(X_data)

        H_xx = H_xx.detach()
        H_yy = H_yy.detach()
        H_bcs = H_bcs.detach()
        H_data = H_data.detach()
        
        f = self.source(X_res).flatten()
        g = self.exact(X_bcs).flatten()

        def fun_residual(coef):
            return self.Nonlinear_system_torch(coef, H_xx, H_yy, H_bcs, H_data, f, g, u_data).view(-1)
        
        N = model.layers[-2]
        beta0 = torch.zeros(N, device=self.device, dtype=self.dtype)
        alpha0 = torch.tensor([0], device=self.device, dtype=self.dtype)
        coef  = torch.cat([alpha0, beta0]).flatten()                    # shape (1+M,) 

        result = least_squares(
                fun=fun_residual,
                x0=coef,                     # 初值张量
                method='trf',                 # 'trf' 稳定
                ftol=1e-12,                    # 放宽收敛阈值
                xtol=1e-12,
                gtol=1e-12,
                tr_solver='exact',            # 小/中规模问题用 exact
                tr_options={'regularize': True},
                max_nfev=200,                 # 最大函数评估次数
                verbose=2
                )

        coef = result.x  # (p,) tensor
        coef = coef.unsqueeze(-1)  # (p,1) 形式

        return coef

    def exact(self, X):
        raise NotImplementedError

    def source(self, X):
        raise NotImplementedError
    
    def IC_fun(self, X_ini):
        return self.exact(X_ini)

    def BC_fun(self, X_bcs):
        return self.exact(X_bcs)

class PoissonInv(PoissonInverse):
    def __init__(self, domain, alpha):
        super().__init__(domain)
        self.beta = alpha

    def exact(self, X):
        u = torch.sin(torch.pi * X[:,0:1]**2) * torch.sin(torch.pi * X[:,1:2]**2) 
        return u.to(self.device)

    def source(self, X):
        x, y = X[:, 0:1], X[:, 1:2]
        f = (2*torch.pi*torch.cos(torch.pi*x**2)*torch.sin(torch.pi*y**2)
            + 2*self.beta*torch.pi*torch.sin(torch.pi*x**2)*torch.cos(torch.pi*y**2)
            - 4*torch.pi**2*(x**2 + self.beta*y**2)
            * torch.sin(torch.pi*x**2)*torch.sin(torch.pi*y**2))
        return f.to(self.device)