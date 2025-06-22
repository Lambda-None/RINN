import torch

####################### 1D Poisson Equations ####################################
class Poisson1DBase:
    def __init__(self, domain, loss_weight_bc=1.0):
        self.domain = domain
        self.device = domain.device
        self.loss_weight_bc = loss_weight_bc

    def build_linear_system(self, model, X_ini, X_bcs, X_res):
        H_bcs = model.get_last_hidden_layer(X_bcs)
        _, H_xx = model.compute_gradients(X_res)

        A = torch.cat([-H_xx, H_bcs], dim=0)
        B = torch.cat([self.source(X_res), self.BC_fun(X_bcs)], dim=0)
        return A, B

    def loss_compute(self, model, X_ini, X_bcs, X_res):
        X_res.requires_grad=True
        X_bcs.requires_grad=True

        f_res = self.source(X_res)
        f_bcs = self.BC_fun(X_bcs)

        u = model.forward(X_res)
        u_x = torch.autograd.grad(u, X_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, X_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_bcs = model.forward(X_bcs)

        loss_equ = torch.sqrt(torch.mean((f_res - (-u_xx))**2))
        loss_bcs = torch.sqrt(torch.mean((f_bcs - u_bcs)**2))
        return loss_equ + self.loss_weight_bc * loss_bcs

    def BC_fun(self, x):
        return self.exact(x)

    def exact(self, x):
        raise NotImplementedError

    def source(self, x):
        raise NotImplementedError

class Poisson1D_1(Poisson1DBase):
    def __init__(self, domain):
        super().__init__(domain)

    def exact(self, x):
        return (torch.sin(2 * torch.pi * x) * torch.cos(3 * torch.pi * x)).to(self.device)

    def source(self, x):
        return (
            13 * torch.pi**2 * torch.sin(2 * torch.pi * x) * torch.cos(3 * torch.pi * x)
            + 12 * torch.pi**2 * torch.cos(2 * torch.pi * x) * torch.sin(3 * torch.pi * x)
        ).to(self.device)

class Poisson1D_k(Poisson1DBase):
    def __init__(self, domain, k):
        super().__init__(domain)
        self.k = k

    def exact(self, x):
        return torch.sin(self.k * torch.pi * x).to(self.device)

    def source(self, x):
        return (self.k**2 * torch.pi**2 * torch.sin(self.k * torch.pi * x)).to(self.device)


####################### 2D Poisson Equations ####################################
class Poisson2DBase:
    def __init__(self, domain):
        self.domain = domain
        self.device = domain.device

    def build_linear_system(self, model, X_ini, X_bcs, X_res):
        H_bcs = model.get_last_hidden_layer(X_bcs)
        _, _, H_xx, H_yy = model.compute_gradients(X_res)

        A = torch.cat([-(H_xx + H_yy), H_bcs], dim=0)
        B = torch.cat([self.source(X_res), self.BC_fun(X_bcs)], dim=0)
        return A, B

    def loss_compute(self, model, X_ini, X_bcs, X_res):
        X_res.requires_grad=True
        X_bcs.requires_grad=True

        u = model.forward(X_res)
        u_bcs = model.forward(X_bcs)
        grad_u = torch.autograd.grad(u, X_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[:, [0]]
        u_y = grad_u[:, [1]]
        u_xx = torch.autograd.grad(u_x, X_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]
        u_yy = torch.autograd.grad(u_y, X_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [1]]

        loss_equ = torch.sqrt(torch.mean(torch.square(self.source(X_res) - (-(u_xx + u_yy)))))
        loss_bcs = torch.sqrt(torch.mean(torch.square(self.BC_fun(X_bcs) - u_bcs)))
        return loss_equ + loss_bcs

    def exact(self, x):
        raise NotImplementedError

    def source(self, x):
        raise NotImplementedError

    def BC_fun(self, x):
        return self.exact(x)
    
class Poisson2D_1(Poisson2DBase):
    def __init__(self, domain):
        super().__init__(domain)

    def exact(self, X):
        return (- (2 * torch.cos((3 / 2) * torch.pi * X[:, 0:1] + 2 * torch.pi / 5) +
                   (3 / 2) * torch.cos(3 * torch.pi * X[:, 0:1] - torch.pi / 5)) *
                (2 * torch.cos((3 / 2) * torch.pi * X[:, 1:2] + 2 * torch.pi / 5) +
                 (3 / 2) * torch.cos(3 * torch.pi * X[:, 1:2] - torch.pi / 5))
               ).to(self.device)

    def source(self, X_res):
        x = X_res[:, 0:1]
        y = X_res[:, 1:2]
        A_x = 2 * torch.cos((3 / 2) * torch.pi * x + 2 * torch.pi / 5) + (3 / 2) * torch.cos(3 * torch.pi * x - torch.pi / 5)
        B_y = 2 * torch.cos((3 / 2) * torch.pi * y + 2 * torch.pi / 5) + (3 / 2) * torch.cos(3 * torch.pi * y - torch.pi / 5)
        term1_Axx = -2 * ((3 * torch.pi / 2)**2) * torch.cos((3 / 2) * torch.pi * x + 2 * torch.pi / 5)
        term2_Axx = -(3 / 2) * (3 * torch.pi)**2 * torch.cos(3 * torch.pi * x - torch.pi / 5)
        A_xx = term1_Axx + term2_Axx
        term1_Byy = -2 * ((3 * torch.pi / 2)**2) * torch.cos((3 / 2) * torch.pi * y + 2 * torch.pi / 5)
        term2_Byy = -(3 / 2) * (3 * torch.pi)**2 * torch.cos(3 * torch.pi * y - torch.pi / 5)
        B_yy = term1_Byy + term2_Byy
        laplacian = (A_xx * B_y + A_x * B_yy)
        return laplacian.to(self.device)

class Poisson2D_4(Poisson2DBase):
    def __init__(self, domain, epsilon):
        super().__init__(domain)
        self.epsilon = epsilon

    def exact(self, X):
        return torch.exp(-(X[:, 0:1]**2 + X[:, 1:2]**2) / self.epsilon).to(self.device)

    def source(self, X_res):
        return (-(4 * (X_res[:, 0:1]**2 + X_res[:, 1:2]**2)) / self.epsilon**2 + 4 / self.epsilon) * \
               torch.exp(-(X_res[:, 0:1]**2 + X_res[:, 1:2]**2) / self.epsilon)


####################### 1D Heat Equations ####################################
class Heat1DBase:
    def __init__(self, domain):
        self.domain = domain
        self.device = domain.device

    def build_linear_system(self, model, X_ini, X_bcs, X_res):
        H_ics = model.get_last_hidden_layer(X_ini)
        H_bcs = model.get_last_hidden_layer(X_bcs)
        _, H_t, H_xx, _ = model.compute_gradients(X_res)

        A = torch.cat([H_t-H_xx, H_bcs, H_ics], dim=0)
        B = torch.cat([self.source(X_res), self.BC_fun(X_bcs), self.IC_fun(X_ini)], dim=0)
        return A, B

    def loss_compute(self, model, X_ini, X_bcs, X_res):
        X_res.requires_grad=True
        X_bcs.requires_grad=True
        X_ini.requires_grad=True

        u = model.forward(X_res)
        u_bcs = model.forward(X_bcs)
        u_ics = model.forward(X_ini)
        grad_u = torch.autograd.grad(u, X_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[:, [0]]
        u_t = grad_u[:, [1]]
        u_xx = torch.autograd.grad(u_x, X_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]

        loss_equ = torch.sqrt(torch.mean(torch.square(self.source(X_res) - (u_t - u_xx))))
        loss_bcs = torch.sqrt(torch.mean(torch.square(self.BC_fun(X_bcs) - u_bcs)))
        loss_ics = torch.sqrt(torch.mean(torch.square(self.IC_fun(X_ini) - u_ics)))
        return loss_equ + loss_bcs + loss_ics

    def exact(self, X):
        raise NotImplementedError

    def source(self, X):
        raise NotImplementedError
    
    def IC_fun(self, X_ini):
        return self.exact(X_ini)

    def BC_fun(self, X_bcs):
        return self.exact(X_bcs)

class Heat1D_k(Heat1DBase):
    def __init__(self, domain, k):
        super().__init__(domain)
        self.k = k

    def exact(self, X):
        u = torch.exp(-X[:,1:2]) * torch.sin(self.k * torch.pi * X[:,0:1]) 
        return u.to(self.device)

    def source(self, X):
        f = (-1 + (self.k * torch.pi)**2) * torch.exp(-X[:,1:2]) * torch.sin(self.k * torch.pi * X[:,0:1]) 
        return f.to(self.device)
    
class Heat1D_1(Heat1DBase):
    def __init__(self, domain):
        super().__init__(domain)

    def exact(self, X):
        u = 1 / (torch.sqrt(1 + 4 * X[:,1:2])) * torch.exp(- X[:,0:1]**2 / (1 + 4 * X[:,1:2]))
        return u.to(self.device)

    def source(self, X):
        f = torch.zeros_like(X[:,1:2])
        return f.to(self.device)


####################### 1D Advection Equations ####################################
class Advection1Dbase:
    def __init__(self, domain, c, IC_fun):
        self.c = c
        self.domain = domain
        self.device = domain.device
        self.IC_fun = IC_fun  

    def build_linear_system(self, model, X_ini, X_bcs, X_res):
        mask_1 = X_bcs[:, 0] == 1
        mask_2 = X_bcs[:, 0] == -1
        X_bc1 = X_bcs[mask_1]
        X_bc2 = X_bcs[mask_2]
        H_ics = model.get_last_hidden_layer(X_ini)
        H_bc1 = model.get_last_hidden_layer(X_bc1)
        H_bc2 = model.get_last_hidden_layer(X_bc2)
        H_x, H_t, _, _ = model.compute_gradients(X_res)

        A = torch.cat([H_t + self.c * H_x, H_bc1 - H_bc2, H_ics], dim=0)
        B = torch.cat([self.source(X_res), self.BC_fun(X_bc1), self.IC_fun(X_ini)], dim=0)
        return A, B

    def loss_compute(self, model, X_ini, X_bcs, X_res):
        X_res.requires_grad=True
        X_bcs.requires_grad=True
        X_ini.requires_grad=True

        mask_1 = X_bcs[:, 0] == 1
        mask_2 = X_bcs[:, 0] == -1
        X_bc1 = X_bcs[mask_1]
        X_bc2 = X_bcs[mask_2]

        u = model.forward(X_res)
        u_bc1 = model.forward(X_bc1)
        u_bc2 = model.forward(X_bc2)
        u_ics = model.forward(X_ini)
        grad_u = torch.autograd.grad(u, X_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[:, [0]]
        u_t = grad_u[:, [1]]

        loss_equ = torch.sqrt(torch.mean(torch.square(self.source(X_res) - (u_t + self.c * u_x))))
        loss_bcs = torch.sqrt(torch.mean(torch.square((u_bc1 - u_bc2))))
        loss_ics = torch.sqrt(torch.mean(torch.square(self.IC_fun(X_ini) - u_ics)))
        return loss_equ + loss_bcs + loss_ics

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
    def __init__(self, domain, c):
        super().__init__(domain, c, self.IC_fun)

    def IC_fun(self, X_ini):
        U_ini = torch.exp(0.5*torch.sin(2 * torch.pi * X_ini[:, 0:1]))-1
        return U_ini

class Advection_2(Advection1Dbase):
    def __init__(self, domain, c):
        super().__init__(domain, c, self.IC_fun)

    def IC_fun(self, X_ini):
        U_ini = torch.sin(2 * torch.pi * X_ini[:, 0:1]) * torch.cos(3 * torch.pi * X_ini[:, 0:1])
        return U_ini


####################### 1D Wave Equations ####################################
class Wave1Dbase:
    def __init__(self, domain, c, IC_fun, IC_grad):
        self.domain = domain
        self.device = domain.device
        self.c = c
        self.IC_fun = IC_fun
        self.IC_grad = IC_grad

    def build_linear_system(self, model, X_ini, X_bcs, X_res):
        # 确保输入数据在正确设备
        X_ini = X_ini.to(self.device)
        X_bcs = X_bcs.to(self.device)
        X_res = X_res.to(self.device)

        # 构建神经网络隐层矩阵与导数项
        H_bcs = model.get_last_hidden_layer(X_bcs)
        H_ics = model.get_last_hidden_layer(X_ini)
        _, _, H_xx, H_tt = model.compute_gradients(X_res)
        _, H_t_ics, _, _ = model.compute_gradients(X_ini)

        # 构造线性系统矩阵 A 和右端项 B
        A = torch.cat([
            H_tt - (self.c ** 2) * H_xx, 
            H_bcs, H_ics, H_t_ics
            ], dim=0)
        B = torch.cat([
            self.source(X_res),
            self.BC_fun(X_bcs),
            self.IC_fun(X_ini),
            self.IC_grad(X_ini)
        ], dim=0)
        return A, B
    
    def loss_compute(self, model, X_ini, X_bcs, X_res):
        X_res.requires_grad=True
        X_bcs.requires_grad=True
        X_ini.requires_grad=True

        u = model.forward(X_res)
        u_bcs = model.forward(X_bcs)
        u_ics = model.forward(X_ini)
        grad_u = torch.autograd.grad(u, X_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad_u[:, [0]]
        u_t = grad_u[:, [1]]
        u_xx = torch.autograd.grad(u_x, X_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]
        u_tt = torch.autograd.grad(u_t, X_res, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, [1]]

        grad_ini = torch.autograd.grad(u_ics, X_ini, grad_outputs=torch.ones_like(u_ics), create_graph=True)[0]
        u_t_ini = grad_ini[:, [1]]

        loss_equ = torch.sqrt(torch.mean(torch.square(self.source(X_res) - (u_tt - (self.c ** 2) * u_xx))))
        loss_bcs = torch.sqrt(torch.mean(torch.square(self.BC_fun(X_bcs) - u_bcs)))
        loss_ics = torch.sqrt(torch.mean(torch.square(self.IC_fun(X_ini) - u_ics)))
        loss_ics_t = torch.sqrt(torch.mean(torch.square(self.IC_grad(X_ini) - u_t_ini)))
        return loss_equ + loss_bcs + loss_ics + loss_ics_t

    def exact(self, X):
        u = torch.zeros_like(X[:, 0:1])
        return u.to(self.device)

    def source(self, X):
        f = torch.zeros_like(X[:, 0:1])
        return f.to(self.device)

    def BC_fun(self, X_bcs):
        U_bcs = torch.zeros_like(X_bcs[:, 1:2])
        return U_bcs

class Wave_1(Wave1Dbase):
    def __init__(self, domain, c):
        super().__init__(domain, c, self.IC_fun, self.IC_grad)

    def IC_fun(self, X_ini):
        U_ini = torch.sin(1 * torch.pi * X_ini[:, 0:1]) \
              + torch.sin(2 * torch.pi * X_ini[:, 0:1]) \
              + torch.sin(3 * torch.pi * X_ini[:, 0:1])
        return U_ini

    def IC_grad(self, X_ini):
        U_ini_t = torch.zeros_like(X_ini[:, 0:1])
        return U_ini_t
    
class Wave_2(Wave1Dbase):
    def __init__(self, domain, c):
        super().__init__(domain, c, self.IC_fun, self.IC_grad)

    def IC_fun(self, X_ini):
        U_ini = torch.sin(2 * torch.pi * X_ini[:, 0:1]) \
              + torch.sin(4 * torch.pi * X_ini[:, 0:1]) 
        return U_ini

    def IC_grad(self, X_ini):
        U_ini_t = torch.zeros_like(X_ini[:, 0:1])
        return U_ini_t
    
class Wave_3(Wave1Dbase):
    def __init__(self, domain, c):
        super().__init__(domain, c, self.IC_fun, self.IC_grad)

    def IC_fun(self, X_ini):
        U_ini = torch.sin(1 * torch.pi * X_ini[:, 0:1]) \
              + torch.sin(3 * torch.pi * X_ini[:, 0:1]) 
        return U_ini

    def IC_grad(self, X_ini):
        U_ini_t = torch.zeros_like(X_ini[:, 0:1])
        return U_ini_t

